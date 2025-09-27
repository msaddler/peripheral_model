import collections

import numpy as np
import scipy.signal
import torch
import torchvision


class PeripheralModel(torch.nn.Module):
    def __init__(
        self,
        sr_input=None,
        sr_output=None,
        config_cochlear_filterbank={},
        config_ihc_transduction={},
        config_ihc_lowpass_filter={},
        config_anf_rate_level={},
        config_anf_spike_generator={},
        config_random_slice={},
    ):
        """
        Construct torch peripheral auditory model from config dictionaries
        """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_input if sr_output is None else sr_output
        self.body = collections.OrderedDict()
        # Bandpass filterbank determines cochlear frequency tuning
        if config_cochlear_filterbank:
            msg = "Cochlear filterbank mode must be `fir_gammatone`"
            assert "fir_gammatone" in config_cochlear_filterbank["mode"], msg
            if config_cochlear_filterbank.get("cfs", False):
                cfs = np.array(config_cochlear_filterbank["cfs"])
            else:
                cfs = erbspace(
                    config_cochlear_filterbank["min_cf"],
                    config_cochlear_filterbank["max_cf"],
                    config_cochlear_filterbank["num_cf"],
                )
            self.body["cochlear_filterbank"] = GammatoneFilterbank(
                sr=sr_input,
                fir_dur=config_cochlear_filterbank.get("fir_dur", 0.05),
                cfs=cfs,
                **config_cochlear_filterbank.get("kwargs_filter_coefs", {}),
            )
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
        # ANF spike generator determines noisiness of spont rate channels
        if config_anf_spike_generator:
            self.body["anf_spike_generator"] = BinomialSpikeGenerator(
                **config_anf_spike_generator,
            )
        self.body = torch.nn.Sequential(self.body)
        # Randomly slice peripheral model representation (trim boundary artifacts)
        if config_random_slice:
            self.head = RandomSlice(**config_random_slice)
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


def freq2erb(freq):
    """
    Convert frequency in Hz to ERB-number.
    Same as `freqtoerb.m` in the AMT.
    """
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb2freq(erb):
    """
    Convert ERB-number to frequency in Hz.
    Same as `erbtofreq.m` in the AMT.
    """
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(freq_min, freq_max, num):
    """
    Create an array of frequencies in Hz evenly spaced on a ERB-number scale.
    Same as `erbspace.m` in the AMT.

    Args
    ----
    freq_min (float): minimum frequency in Hz
    freq_max (float): maximum frequency Hz
    num (int): number of frequencies (length of array)

    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    """
    return erb2freq(np.linspace(freq2erb(freq_min), freq2erb(freq_max), num=num))


def get_gammatone_filter_coefs(sr, cfs, EarQ=9.2644, minBW=24.7, order=1):
    """
    Based on `MakeERBFilters.m` and `ERBFilterBank.m`
    from Malcolm Slaney's Auditory Toolbox (1998).
    """
    T = 1 / sr
    ERB = ((cfs / EarQ) ** order + minBW**order) ** (1 / order)
    B = 1.019 * 2 * np.pi * ERB
    A0 = T * np.ones_like(cfs)
    A2 = 0 * np.ones_like(cfs)
    B0 = 1 * np.ones_like(cfs)
    B1 = -2 * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    tmp0 = 2 * T * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    tmp1 = T * np.sin(2 * cfs * np.pi * T) / np.exp(B * T)
    A11 = -(tmp0 + 2 * np.sqrt(3 + 2**1.5) * tmp1) / 2
    A12 = -(tmp0 - 2 * np.sqrt(3 + 2**1.5) * tmp1) / 2
    A13 = -(tmp0 + 2 * np.sqrt(3 - 2**1.5) * tmp1) / 2
    A14 = -(tmp0 - 2 * np.sqrt(3 - 2**1.5) * tmp1) / 2

    tmp2 = np.exp(4 * 1j * cfs * np.pi * T)
    tmp3 = 2 * np.exp(-(B * T) + 2 * 1j * cfs * np.pi * T) * T
    tmp4 = np.cos(2 * cfs * np.pi * T)
    tmp5 = np.sin(2 * cfs * np.pi * T)
    gain = np.abs(
        (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 - 2 ** (3 / 2)) * tmp5))
        * (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 - 2 ** (3 / 2)) * tmp5))
        * (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 + 2 ** (3 / 2)) * tmp5))
        * (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 + 2 ** (3 / 2)) * tmp5))
        / (-2 / np.exp(2 * B * T) - 2 * tmp2 + 2 * (1 + tmp2) / np.exp(B * T)) ** 4
    )

    filter_coefs = [
        {"b": np.array([A0, A11, A2]) / gain, "a": np.array([B0, B1, B2])},
        {"b": np.array([A0, A12, A2]), "a": np.array([B0, B1, B2])},
        {"b": np.array([A0, A13, A2]), "a": np.array([B0, B1, B2])},
        {"b": np.array([A0, A14, A2]), "a": np.array([B0, B1, B2])},
    ]
    return filter_coefs


def scipy_gammatone_filterbank(x, filter_coefs):
    """
    Convert signal waveform `x` to set of subbands `x_subbands`
    using scipy.signal.lfilter and the gammatone filterbank
    instantiated by `filter_coefs`.
    """
    if len(x.shape) == 1:
        x_subbands = x[np.newaxis, np.newaxis, :]
    elif len(x.shape) == 2:
        x_subbands = x[:, np.newaxis, :]
    else:
        raise ValueError("Expected input shape [time] or [batch, time]")
    n_subbands = filter_coefs[0]["b"].shape[-1]
    x_subbands = np.tile(x_subbands, [1, n_subbands, 1])
    for fc in filter_coefs:
        for itr_subbands in range(n_subbands):
            x_subbands[:, itr_subbands, :] = scipy.signal.lfilter(
                fc["b"][:, itr_subbands],
                fc["a"][:, itr_subbands],
                x_subbands[:, itr_subbands, :],
                axis=-1,
            )
    if len(x.shape) == 1:
        x_subbands = x_subbands[0]
    return x_subbands


def get_gammatone_impulse_responses(sr, fir_dur, cfs, EarQ=9.2644, minBW=24.7, order=1):
    """ """
    impulse = np.zeros(int(fir_dur * sr))
    impulse[0] = 1
    filter_coefs = get_gammatone_filter_coefs(
        sr, cfs, EarQ=EarQ, minBW=minBW, order=order
    )
    impulse_responses = scipy_gammatone_filterbank(impulse, filter_coefs)
    return impulse_responses


def ihc_lowpass_filter_fir(sr, fir_dur, cutoff=3e3, order=7):
    """
    Returns finite response of IHC lowpass filter from
    bez2018model/model_IHC_BEZ2018.c
    """
    n_taps = int(sr * fir_dur)
    if n_taps % 2 == 0:
        n_taps = n_taps + 1
    impulse = np.zeros(n_taps)
    impulse[0] = 1
    fir = np.zeros(n_taps)
    ihc = np.zeros(order + 1)
    ihcl = np.zeros(order + 1)
    c1LP = (sr - 2 * np.pi * cutoff) / (sr + 2 * np.pi * cutoff)
    c2LP = (np.pi * cutoff) / (sr + 2 * np.pi * cutoff)
    for n in range(n_taps):
        ihc[0] = impulse[n]
        for i in range(order):
            ihc[i + 1] = (c1LP * ihcl[i + 1]) + c2LP * (ihc[i] + ihcl[i])
        ihcl = ihc
        fir[n] = ihc[order]
    fir = fir * scipy.signal.windows.hann(n_taps)
    fir = fir / fir.sum()
    return fir


class FIRFilterbank(torch.nn.Module):
    def __init__(self, fir, dtype=torch.float32, **kwargs_conv1d):
        """
        FIR filterbank

        Args
        ----
        fir (list or np.ndarray or torch.Tensor):
            Filter coefficients. Shape (n_taps,) or (n_filters, n_taps)
        dtype (torch.dtype):
            Data type to cast `fir` to in case it is not a `torch.Tensor`
        kwargs_conv1d (kwargs):
            Keyword arguments passed on to torch.nn.functional.conv1d
            (must not include `groups`, which is used for batching)
        """
        super().__init__()
        if not isinstance(fir, (list, np.ndarray, torch.Tensor)):
            raise TypeError(
                "fir must be list, np.ndarray or torch.Tensor, got "
                f"{fir.__class__.__name__}"
            )
        if isinstance(fir, (list, np.ndarray)):
            fir = torch.tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            raise ValueError(
                "fir must be one- or two-dimensional with shape (n_taps,) or "
                f"(n_filters, n_taps), got shape {fir.shape}"
            )
        self.register_buffer("fir", fir)
        self.kwargs_conv1d = kwargs_conv1d

    def forward(self, x, batching=False):
        """
        Filter input signal

        Args
        ----
        x (torch.Tensor): Input signal
        batching (bool):
            If `True`, the input is assumed to have shape (..., n_filters, time)
            and each channel is filtered with its own filter

        Returns
        -------
        y (torch.Tensor): Filtered signal
        """
        y = x
        if batching:
            assert y.shape[-2] == self.fir.shape[0]
        else:
            y = y.unsqueeze(-2)
        unflatten_shape = y.shape[:-2]
        y = torch.flatten(y, start_dim=0, end_dim=-2 - 1)
        y = torch.nn.functional.conv1d(
            input=torch.nn.functional.pad(y, (self.fir.shape[-1] - 1, 0)),
            weight=self.fir.flip(-1).view(-1, 1, self.fir.shape[-1]),
            **self.kwargs_conv1d,
            groups=y.shape[-2] if batching else 1,
        )
        y = torch.unflatten(y, 0, unflatten_shape)
        if self.fir.ndim == 1:
            y = y.squeeze(-2)
        return y


class GammatoneFilterbank(torch.nn.Module):
    def __init__(
        self,
        sr=20e3,
        fir_dur=0.05,
        cfs=erbspace(8e1, 8e3, 50),
        dtype=torch.float32,
        **kwargs,
    ):
        """ """
        super().__init__()
        fir = get_gammatone_impulse_responses(
            sr=sr,
            fir_dur=fir_dur,
            cfs=cfs,
            **kwargs,
        )
        self.fb = FIRFilterbank(fir, dtype=dtype)

    def forward(self, x, batching=False):
        """ """
        return self.fb(x, batching=batching)


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


class IHCLowpassFilter(FIRFilterbank):
    def __init__(
        self,
        sr_input=20e3,
        sr_output=10e3,
        fir_dur=0.05,
        cutoff=3e3,
        order=7,
        dtype=torch.float32,
    ):
        """ """
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


class Hilbert(torch.nn.Module):
    def __init__(self, dim=-1):
        """
        Compute the analytic signal, using the Hilbert transform
        (torch implementation of `scipy.signal.hilbert`)
        """
        super().__init__()
        self.dim = dim

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        """ """
        n = x.shape[self.dim]
        X = torch.fft.fft(x, n=n, dim=self.dim, norm=None)
        h = torch.zeros(n, dtype=X.dtype).to(X.device)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2
        ind = [np.newaxis] * x.ndim
        ind[self.dim] = slice(None)
        return torch.fft.ifft(
            X * h[ind],
            n=n,
            dim=self.dim,
            norm=None,
        )


class HilbertEnvelope(torch.nn.Module):
    def __init__(self, **args):
        """ """
        super().__init__()
        self.hilbert = Hilbert(**args)

    def forward(self, x):
        return torch.abs(self.hilbert(x))


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
        """ """
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
        assert np.all(rate_max > rate_spont), "rate_max must be greater than rate_spont"
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
        self.register_buffer(
            "rate_spont", torch.tensor(rate_spont, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "rate_max", torch.tensor(rate_max, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "threshold", torch.tensor(threshold, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "dynamic_range", torch.tensor(dynamic_range, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "dynamic_range_interval",
            torch.tensor(dynamic_range_interval, dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "y_threshold", torch.tensor(y_threshold, dtype=dtype), persistent=True
        )
        self.register_buffer("k", torch.tensor(k, dtype=dtype), persistent=True)
        self.register_buffer("x0", torch.tensor(x0, dtype=dtype), persistent=True)
        # Construct envelope extraction function if needed
        self.envelope_mode = envelope_mode
        if self.envelope_mode:
            self.envelope_function = HilbertEnvelope(dim=-1)

    def resize(self, x, shape):
        """ """
        x = np.array(x).reshape([-1])
        if len(x) == 1:
            x = np.full(shape, x[0])
        else:
            x = np.reshape(x, shape)
        return x

    def forward(self, tensor_subbands):
        """ """
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
            tensor_pa = tensor_pa ** self.compression_power.view(1, 1, -1, 1)
        # Compute sigmoid function with tensor broadcasting
        x = 20.0 * torch.log(tensor_pa / 20e-6) / np.log(10)
        y = 1.0 / (1.0 + torch.exp(-self.k * (x - self.x0)))
        if self.envelope_mode:
            y = y * tensor_tfs
        tensor_rates = self.rate_spont + (self.rate_max - self.rate_spont) * y
        return tensor_rates


class BinomialSpikeGenerator(torch.nn.Module):
    def __init__(
        self,
        sr=10000,
        mode="approx",
        n_per_channel=[384, 160, 96],
        n_per_step=48,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        self.sr = sr
        self.mode = mode
        self.n_per_step = n_per_step
        self.register_buffer(
            "n_per_channel",
            torch.tensor(n_per_channel, dtype=dtype).view([-1]),
            persistent=True,
        )

    def forward(self, tensor_rates):
        """ """
        msg = "Requires input shape [batch, channel, freq, time]"
        assert tensor_rates.ndim == 4, msg
        tensor_probs = tensor_rates / self.sr
        if self.mode == "approx":
            # Sample from normal approximation of binomial distribution
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            sample = torch.distributions.normal.Normal(
                loc=n * p,
                scale=torch.sqrt(n * p * (1 - p)),
                validate_args=False,
            ).rsample()
            tensor_spike_counts = torch.round(torch.nn.functional.relu(sample))
        elif self.mode == "exact":
            # Binomial distribution implemented as sum of Bernoulli random variables
            n = self.n_per_channel
            p = tensor_probs
            assert (n.ndim == 1) and (n.shape[0] == p.shape[1])
            tensor_spike_counts = torch.zeros_like(p)
            for channel in range(p.shape[1]):
                total = int(n[channel])
                count = 0
                while count < total:
                    n_sample_per_step = min(self.n_per_step, total - count)
                    sample = (
                        torch.rand(
                            size=(n_sample_per_step, *p[:, channel, :, :].shape),
                            device=self.n_per_channel.device,
                        )
                        < p[None, :, channel, :, :]
                    )
                    tensor_spike_counts[:, channel, :, :] += sample.sum(dim=0)
                    count += n_sample_per_step
        elif self.mode == "additive":
            # Replace sampling with additive noise to enable back-propagation
            n = self.n_per_channel.view([1, -1, 1, 1])
            p = tensor_probs
            noise = torch.randn_like(p) / n
            tensor_spike_counts = torch.nn.functional.relu((p + noise) * n)
        else:
            raise NotImplementedError(f"mode=`{self.mode}` is not implemented")
        return tensor_spike_counts


class RandomSlice(torch.nn.Module):
    def __init__(self, size=[50, 20000], buffer=[0, 0], **kwargs):
        """ """
        super().__init__()
        self.size = size
        self.pre_crop_slice = []
        for b in buffer:
            if b is None:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, int) and b > 0:
                self.pre_crop_slice.append(slice(b, -b))
            elif isinstance(b, int) and b == 0:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, (tuple, list)):
                self.pre_crop_slice.append(slice(*b))
        self.crop = torchvision.transforms.RandomCrop(size=self.size, **kwargs)

    def forward(self, x):
        """ """
        return self.crop(x[..., *self.pre_crop_slice])
