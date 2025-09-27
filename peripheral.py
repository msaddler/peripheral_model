import collections
import copy
import math

import numpy as np
import scipy.signal
import soxr
import torch

import util


class Model(torch.nn.Module):
    def __init__(
        self,
        sr_input=None,
        sr_output=None,
        config_middle_ear_filter={},
        config_cochlear_filterbank={},
        config_ihc_transduction={},
        config_ihc_lowpass_filter={},
        config_anf_rate_level={},
        config_interpolate={},
        config_anf_spike_generator={},
        config_random_slice={},
    ):
        """
        Construct torch peripheral auditory model from config dictionaries.
        """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_input if sr_output is None else sr_output
        self.body = collections.OrderedDict()
        # Middle ear filter (FIR filter convolved with input sounds)
        if config_middle_ear_filter:
            self.body["middle_ear_filter"] = MiddleEarFilter(
                sr=self.sr_input,
                **config_middle_ear_filter,
            )
        # Bandpass filterbank determines cochlear frequency tuning
        if config_cochlear_filterbank:
            args = copy.deepcopy(config_cochlear_filterbank)
            if isinstance(args["cfs"], dict):
                args["cfs"] = util.erbspace(**args["cfs"])
            filterbank_type = args.pop("type")
            if filterbank_type == "gammatone_filterbank_fir":
                self.body["cochlear_filterbank"] = GammatoneFilterbank(
                    sr=sr_input,
                    **args,
                )
            else:
                raise ValueError(f"{filterbank_type=} not recognized")
        else:
            self.body["cochlear_filterbank"] = torch.nn.Identity()
        # IHC transduction (includes compression and half-wave rectification)
        if config_ihc_transduction:
            self.body["ihc_transduction"] = IHCTransduction(
                **config_ihc_transduction,
            )
        # IHC lowpass filter determines phase locking limit
        if config_ihc_lowpass_filter:
            self.body["ihc_lowpass_filter"] = IHCLowpassFilter(
                sr_input=self.sr_input,
                sr_output=self.sr_output,
                **config_ihc_lowpass_filter,
            )
        # Rate-level function determines thresholds and dynamic ranges
        if config_anf_rate_level:
            self.body["anf_rate_level"] = SigmoidRateLevelFunction(
                **config_anf_rate_level,
            )
        # Interpolate firing rates to enable multiple IHCs per CF
        if config_interpolate:
            self.body["interpolate"] = util.Interpolate(
                **config_interpolate,
            )
        # ANF spike generator determines noisiness of spont rate channels
        if config_anf_spike_generator:
            self.body["anf_spike_generator"] = BinomialSpikeGenerator(
                **config_anf_spike_generator,
            )
        self.body = torch.nn.Sequential(self.body)
        # Randomly slice peripheral model representation (trim boundary artifacts)
        if config_random_slice:
            self.head = util.RandomSlice(**config_random_slice)
        else:
            self.head = torch.nn.Identity()

    def forward(self, x):
        """ """
        if x.shape[-1] == 2:
            assert x.ndim in [3, 5], "expected binaural audio or nervegram input"
            y0 = self.body(x[..., 0])
            y1 = self.body(x[..., 1])
            if y0.ndim == 4:
                # Concatenate peripheral auditory representations along axis 1
                y = torch.concat([y0, y1], axis=1)
            else:
                # If output is audio, preserve format by stacking along axis -1
                y = torch.stack([y0, y1], axis=-1)
        else:
            y = self.body(x)
        y = self.head(y)
        return y


def middle_ear_filter_fir(
    sr,
    mode="iso226_2003",
    numtaps=513,
    minimum_phase=True,
):
    """
    Returns impulse response of a middle ear filter.

    Args
    ----
    mode (str): specifies implementation (e.g., `iso226_2003`)
    numtaps (int): FIR length must be an odd number
    minimum_phase (bool): return asymmetric minimum-phase FIR filter

    Returns
    -------
    fir (np.ndarray): finite impulse response with shape (numptaps,)
    """
    if mode.lower() == "iso226_2003":
        frequency_hz, threshold_dbspl = util.data_iso226_2003()
        freq = [0.0] + list(frequency_hz)
        gain = [0.0] + list(np.power(10, -threshold_dbspl / 20))
    else:
        raise ValueError(f"{mode=} not recognized")
    assert sr / 2 >= freq[-1], f"sr must be >= {2 * freq[-1]} Hz"
    assert numtaps % 2 == 1, f"{numtaps=} must be an odd number"
    for f in np.arange(freq[-1] + 1e3, sr / 2, 1e3):
        freq.append(f)
        gain.append(gain[-1] / 1.1)
    if freq[-1] < sr / 2:
        f = sr / 2
        g = gain[-1] / (1 + (f - freq[-1]) * 0.1 / 1e3)
        freq.append(f)
        gain.append(g)
    freq = np.array(freq)
    gain = np.array(gain)
    fir = scipy.signal.firwin2(
        numtaps=numtaps,
        freq=freq,
        gain=gain,
        nfreqs=None,
        window="hamming",
        antisymmetric=False,
        fs=sr,
    )
    if minimum_phase:
        fir = np.fft.fft(fir)
        fir = np.abs(fir) * np.exp(-1j * scipy.signal.hilbert(np.log(np.abs(fir))).imag)
        fir = np.fft.ifft(fir).real
    return fir


class MiddleEarFilter(util.FIRFilterbank):
    def __init__(
        self,
        sr=20e3,
        mode="iso226_2003",
        numtaps=513,
        minimum_phase=True,
        dtype=torch.float32,
    ):
        """ """
        fir = middle_ear_filter_fir(
            sr=sr,
            mode=mode,
            numtaps=numtaps,
            minimum_phase=minimum_phase,
        )
        super().__init__(fir, dtype=dtype)


def gammatone_filterbank_fir(
    sr,
    cfs,
    fir_dur=0.05,
    order=4,
    bw_mult=None,
):
    """
    Returns impulse responses of a Gammatone filter bank.

    Args
    ----
    sr (float): Sampling rate in Hz
    fir_dur (float): Duration of FIR in seconds
    cfs (float or np.ndarray): Center frequencies with shape (n_filters,)
    order (int):  Filter order
    bw_mult (float or np.ndarray or None): Bandwidth scaling factor

    Returns
    -------
    fir (np.ndarray): impulse responses with shape (n_filters, int(sr * fir_dur))
    """
    cfs = np.array(cfs).reshape([-1])
    if bw_mult is None:
        bw_mult = np.divide(
            math.factorial(order - 1) ** 2,
            (np.pi * math.factorial(2 * order - 2) * 2 ** (-2 * order + 2)),
        )
    else:
        bw_mult = np.array(bw_mult)
    bw = 2 * np.pi * bw_mult * (24.7 + cfs / 9.265)
    wc = 2 * np.pi * cfs
    t = np.arange(0, fir_dur, 1 / sr)
    a = (
        2
        / math.factorial(order - 1)
        / np.abs(1 / bw**order + 1 / (bw + 2j * wc) ** order)
        / sr
    )
    fir = (
        a[:, None]
        * t ** (order - 1)
        * np.exp(-bw[:, None] * t[None, :])
        * np.cos(wc[:, None] * t[None, :])
    )
    return fir


class GammatoneFilterbank(util.FIRFilterbank):
    def __init__(
        self,
        sr=20e3,
        fir_dur=0.05,
        cfs=util.erbspace(8e1, 8e3, 50),
        dtype=torch.float32,
        **kwargs,
    ):
        """
        Gammatone cochlear filterbank, applied by convolution
        with a set of finite impulse responses.
        """
        fir = gammatone_filterbank_fir(
            sr=sr,
            fir_dur=fir_dur,
            cfs=cfs,
            **kwargs,
        )
        super().__init__(fir, dtype=dtype)


class IHCTransduction(torch.nn.Module):
    def __init__(
        self,
        compression_power=None,
        compression_dbspl_min=None,
        compression_dbspl_max=None,
        rectify=True,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        if compression_power is not None:
            self.register_buffer(
                "compression_power",
                torch.tensor(compression_power, dtype=dtype),
            )
        else:
            self.compression_power = None
        if compression_dbspl_min is not None:
            self.compression_pa_min = torch.tensor(
                20e-6 * np.power(10, compression_dbspl_min / 20),
                dtype=dtype,
            )
        else:
            self.compression_pa_min = torch.tensor(-np.inf, dtype=dtype)
        if compression_dbspl_max is not None:
            self.compression_pa_max = torch.tensor(
                20e-6 * np.power(10, compression_dbspl_max / 20),
                dtype=dtype,
            )
        else:
            self.compression_pa_max = torch.tensor(np.inf, dtype=dtype)
        self.rectify = rectify

    def forward(self, x):
        """ """
        if self.compression_power is not None:
            # Broken-stick compression (power compression between
            # compression_dbspl_min and compression_dbspl_max)
            if self.compression_power.ndim > 0:
                if not self.compression_power.ndim == x.ndim:
                    shape = [1 for _ in range(x.ndim)]
                    shape[-2] = x.shape[-2]
                    self.compression_power = self.compression_power.view(*shape)
            abs_x = torch.abs(x)
            IDX_COMPRESSION = torch.logical_and(
                abs_x >= self.compression_pa_min,
                abs_x < self.compression_pa_max,
            )
            IDX_AMPLIFICATION = abs_x < self.compression_pa_min
            x = torch.sign(x) * torch.where(
                IDX_COMPRESSION,
                abs_x**self.compression_power,
                torch.where(
                    IDX_AMPLIFICATION,
                    abs_x * (self.compression_pa_min ** (self.compression_power - 1)),
                    abs_x,
                ),
            )
        if self.rectify:
            # Half-wave rectification
            x = torch.nn.functional.relu(x, inplace=False)
        return x


def ihc_lowpass_filter_fir(sr, fir_dur, cutoff=3e3, order=7):
    """
    Returns finite impulse response of IHC lowpass filter from
    bez2018model/model_IHC_BEZ2018.c
    """
    sr_bez2018 = 100e3

    def resample(x):
        return soxr.resample(
            x,
            in_rate=sr_bez2018,
            out_rate=sr,
            quality="VHQ",
        )

    n_taps = int(sr_bez2018 * fir_dur)
    while len(resample(np.zeros(n_taps))) % 2 == 0:
        n_taps = n_taps + 1
    impulse = np.zeros(n_taps)
    impulse[0] = 1
    fir = np.zeros(n_taps)
    ihc = np.zeros(order + 1)
    ihcl = np.zeros(order + 1)
    c1LP = (sr_bez2018 - 2 * np.pi * cutoff) / (sr_bez2018 + 2 * np.pi * cutoff)
    c2LP = (np.pi * cutoff) / (sr_bez2018 + 2 * np.pi * cutoff)
    for n in range(n_taps):
        ihc[0] = impulse[n]
        for i in range(order):
            ihc[i + 1] = (c1LP * ihcl[i + 1]) + c2LP * (ihc[i] + ihcl[i])
        ihcl = ihc
        fir[n] = ihc[order]
    fir = fir * scipy.signal.windows.hann(n_taps)
    fir = resample(fir)
    fir = fir / fir.sum()
    return fir


class IHCLowpassFilter(util.FIRFilterbank):
    def __init__(
        self,
        sr_input=20e3,
        sr_output=10e3,
        fir_dur=0.05,
        cutoff=3e3,
        order=7,
        dtype=torch.float32,
    ):
        """
        Inner hair cell low-pass filter, applied by convolution
        with a finite impulse response from BEZ2018 model.
        """
        fir = ihc_lowpass_filter_fir(
            sr=sr_input,
            fir_dur=fir_dur,
            cutoff=cutoff,
            order=order,
        )
        stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(stride, sr_input / sr_output), msg
        super().__init__(fir, dtype=dtype, stride=stride)


class SigmoidRateLevelFunction(torch.nn.Module):
    def __init__(
        self,
        rate_spont=[0.0, 0.0, 0.0],
        rate_max=[250.0, 250.0, 250.0],
        threshold=[0.0, 12.0, 28.0],
        dynamic_range=[20.0, 40.0, 80.0],
        dynamic_range_interval=0.95,
        compression_power=None,
        compression_power_default=0.3,
        envelope_mode=True,
        dtype=torch.float32,
    ):
        """
        Sigmoid function to convert sound pressure in Pa to auditory nerve firing
        rates in spikes per second. This function can incorporate a compressive
        nonlinearity and crudely account for audibility and saturation limits.
        List arguments are used to specify parameters for different spontaneous
        rate channels (high, medium, and low spontaneous rate fiber types).

        Args
        ----
        rate_spont (list): spontaneous firing rates in spikes/s
        rate_max (list): maximum firing rates in spikes/s
        threshold (list): auditory nerve fiber thresholds for spiking (dB SPL)
        dynamic_range (list): dynamic ranges over which firing rate changes (dB)
        dynamic_range_interval (float): determines proportion of firing rate change
            within dynamic_range (default is 95%)
        compression_power_default (float): compression_power used to set rate-level
            function parameters (use 0.3 to model normal hearing)
        compression_power (float or frequency-specific array): compression_power
            to apply to inputs (range: 0.3 = normal to 1.0 = linearized)
        envelope_mode (bool): if True, apply compression to envelopes (not TFS)
        dtype (torch.dtype): data type for inputs and internal tensors
        """
        super().__init__()
        if compression_power is not None:
            # Explicitly incorporate power compression into the rate-level function
            self.register_buffer(
                "compression_power",
                torch.tensor(compression_power, dtype=dtype),
            )
            if compression_power_default is not None:
                # Adjust threshold and dynamic range for `compression_power_default`
                shift = 20 * np.log10(20e-6 ** (compression_power_default - 1))
                threshold = np.array(threshold) * compression_power_default + shift
                dynamic_range = np.array(dynamic_range) * compression_power_default
        else:
            self.compression_power = None
        # Check arguments and register tensors with channel-specific shapes
        msg = "rate_max must be greater than rate_spont for each channel"
        assert np.all(np.array(rate_max) > np.array(rate_spont)), msg
        argument_lengths = [
            len(rate_spont),
            len(rate_max),
            len(threshold),
            len(dynamic_range),
        ]
        channel_specific_size = [1, max(argument_lengths), 1, 1]
        rate_spont = self.resize(rate_spont, channel_specific_size)
        rate_max = self.resize(rate_max, channel_specific_size)
        threshold = self.resize(threshold, channel_specific_size)
        dynamic_range = self.resize(dynamic_range, channel_specific_size)
        y_threshold = (1 - dynamic_range_interval) / 2
        k = np.log((1 / y_threshold) - 1) / (dynamic_range / 2)
        x0 = threshold - (np.log((1 / y_threshold) - 1) / (-k))
        self.register_buffer("rate_spont", torch.tensor(rate_spont, dtype=dtype))
        self.register_buffer("rate_max", torch.tensor(rate_max, dtype=dtype))
        self.register_buffer("threshold", torch.tensor(threshold, dtype=dtype))
        self.register_buffer("dynamic_range", torch.tensor(dynamic_range, dtype=dtype))
        self.register_buffer(
            "dynamic_range_interval",
            torch.tensor(dynamic_range_interval, dtype=dtype),
        )
        self.register_buffer("y_threshold", torch.tensor(y_threshold, dtype=dtype))
        self.register_buffer("k", torch.tensor(k, dtype=dtype))
        self.register_buffer("x0", torch.tensor(x0, dtype=dtype))
        # Construct envelope extraction function if needed
        self.envelope_mode = envelope_mode
        if self.envelope_mode:
            self.envelope_function = util.HilbertEnvelope(dim=-1)

    def resize(self, x, shape):
        """ """
        x = np.array(x).reshape([-1])
        if len(x) == 1:
            x = np.full(shape, x[0])
        else:
            x = np.reshape(x, shape)
        return x

    def forward(self, tensor_subbands):
        """
        Apply sigmoid auditory nerve rate-level function.

        Args
        ----
        tensor_subbands (torch.Tensor): half-wave rectified subbands
            with shape [batch, freq, time]

        Returns
        -------
        tensor_rates (torch.Tensor): instantaneous auditory nerve firing rates
            with shape [batch, spont, freq, time]
        """
        while tensor_subbands.ndim < 4:
            tensor_subbands = tensor_subbands.unsqueeze(-3)
        if self.envelope_mode:
            # Subband envelopes are passed through sigmoid and recombined with TFS
            tensor_env = self.envelope_function(tensor_subbands)
            tensor_tfs = torch.divide(tensor_subbands, tensor_env)
            tensor_tfs = torch.where(
                torch.isfinite(tensor_tfs), tensor_tfs, tensor_subbands
            )
            tensor_pa = tensor_env
        else:
            # Subbands are passed through sigmoid (alters spike timing at high levels)
            tensor_pa = tensor_subbands
        if self.compression_power is not None:
            # Apply power compression (supports frequency-specific power compression)
            tensor_pa = util.GradientClippedPower.apply(
                tensor_pa,
                self.compression_power.view(1, 1, -1, 1),
                1.0,
            )
        # Compute sigmoid function with tensor broadcasting
        x = 20.0 * torch.log(tensor_pa / 20e-6) / np.log(10)
        y = util.GradientStableSigmoid.apply(
            x,
            self.k,
            self.x0,
        )
        if self.envelope_mode:
            y = torch.nn.functional.relu(y * tensor_tfs)
        tensor_rates = self.rate_spont + (self.rate_max - self.rate_spont) * y
        return tensor_rates


class BinomialSpikeGenerator(torch.nn.Module):
    def __init__(
        self,
        sr=10000,
        mode="approx",
        n_per_channel=[384, 160, 96],
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        self.sr = sr
        self.mode = mode
        self.register_buffer(
            "n_per_channel",
            torch.tensor(n_per_channel, dtype=dtype).view([-1]),
            persistent=True,
        )

    def forward(self, tensor_rates):
        """ """
        msg = "Requires input shape [batch, channel, freq, time]"
        assert tensor_rates.ndim == 4, msg
        tensor_probs = torch.clamp(tensor_rates / self.sr, min=0, max=1)
        if self.mode == "approx":
            # Sample from normal approximation of binomial distribution
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            sample = torch.distributions.normal.Normal(
                loc=n * p,
                scale=torch.sqrt(n * p * (1 - p)),
                validate_args=False,
            ).rsample()
            sample = torch.nn.functional.relu(sample)
            tensor_spike_counts = sample + sample.round().detach() - sample.detach()
        elif self.mode == "exact":
            # Binomial distribution implemented as sum of Bernoulli random variables
            n = self.n_per_channel
            p = tensor_probs
            assert (n.ndim == 1) and (n.shape[0] == p.shape[1])
            tensor_spike_counts = torch.zeros_like(p)
            for channel in range(p.shape[1]):
                for _ in range(int(n[channel])):
                    tensor_spike_counts[:, channel, :, :] += torch.bernoulli(
                        p[:, channel, :, :]
                    )
        elif self.mode == "additive":
            # Replace sampling with additive noise to enable back-propagation
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            noise = torch.randn_like(p) / n
            tensor_spike_counts = torch.nn.functional.relu((p + noise) * n)
        else:
            raise NotImplementedError(f"mode=`{self.mode}` is not implemented")
        return tensor_spike_counts
