import numpy as np
import scipy.signal
import torch
import torchvision


def format_axes(
    ax,
    str_title=None,
    str_xlabel=None,
    str_ylabel=None,
    fontsize_title=12,
    fontsize_labels=12,
    fontsize_ticks=12,
    fontweight_title=None,
    fontweight_labels=None,
    xscale="linear",
    yscale="linear",
    xlimits=None,
    ylimits=None,
    xticks=None,
    yticks=None,
    xticks_minor=None,
    yticks_minor=None,
    xticklabels=None,
    yticklabels=None,
    spines_to_hide=[],
    major_tick_params_kwargs_update={},
    minor_tick_params_kwargs_update={},
):
    """
    Helper function for setting axes-related formatting parameters.
    """
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)

    major_tick_params_kwargs = {
        "axis": "both",
        "which": "major",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 2,
        "direction": "out",
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)

    minor_tick_params_kwargs = {
        "axis": "both",
        "which": "minor",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 4,
        "direction": "out",
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)

    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)

    return ax


def rms(x):
    """
    Returns root mean square amplitude of x (raises ValueError if NaN).
    """
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError("rms calculation resulted in NaN")
    return out


def get_dbspl(x, mean_subtract=True):
    """
    Returns sound pressure level of x in dB re 20e-6 Pa (dB SPL).
    """
    if mean_subtract:
        x = x - np.mean(x)
    out = 20 * np.log10(rms(x) / 20e-6)
    return out


def set_dbspl(x, dbspl, mean_subtract=True):
    """
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa.
    """
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dbspl / 20)
    return rms_out * x / rms(x)


def combine_signal_and_noise(signal, noise, snr, mean_subtract=True):
    """
    Adds noise to signal with the specified signal-to-noise ratio (snr).
    If snr is finite, the noise waveform is rescaled and added to the
    signal waveform. If snr is positive infinity, returned waveform is
    equal to the signal waveform. If snr is negative inifinity, returned
    waveform is equal to the noise waveform.

    Args
    ----
    signal (np.ndarray): signal waveform
    noise (np.ndarray): noise waveform
    snr (float): signal-to-noise ratio in dB
    mean_subtract (bool): if True, signal and noise are first de-meaned
        (mean_subtract=True is important for accurate snr computation)

    Returns
    -------
    signal_and_noise (np.ndarray or tuple) combined signal and noise
        waveform if input `signal` and `noise` have the same shape. If
        not, signal_and_noise is a tuple: (signal, rescaled_noise)
    """
    if mean_subtract:
        signal = signal - np.mean(signal)
        noise = noise - np.mean(noise)
    if np.isinf(snr) and snr > 0:
        signal_and_noise = (signal, np.zeros_like(noise))
    elif np.isinf(snr) and snr < 0:
        signal_and_noise = (np.zeros_like(signal), noise)
    else:
        rms_noise_scaling = rms(signal) / (rms(noise) * np.power(10, snr / 20))
        signal_and_noise = (signal, rms_noise_scaling * noise)
    if signal_and_noise[0].shape == signal_and_noise[1].shape:
        signal_and_noise = signal_and_noise[0] + signal_and_noise[1]
    return signal_and_noise


def periodogram(x, sr, db=True, p_ref=20e-6, scaling="spectrum", **kwargs):
    """
    Compute power spectrum (default) or power spectral density of signal.

    Args
    ----
    x (np.ndarray): input waveform (Pa)
    sr (int): sampling rate (Hz)
    db (bool): convert output to dB
    p_ref (float): reference pressure for dB conversion (20e-6 Pa for dB SPL)
    scaling (str): "spectrum" (units Pa^2) or "density" (units Pa^2 / Hz)
    kwargs (keyword arguments): passed directly to scipy.signal.periodogram

    Returns
    -------
    fxx (np.ndarray): frequency vector (Hz)
    pxx (np.ndarray): Power spectrum (dB) or power spectral density (dB / Hz)
    """
    fxx, pxx = scipy.signal.periodogram(x=x, fs=sr, scaling=scaling, **kwargs)
    if db:
        p_ref = 1.0 if p_ref is None else p_ref
        pxx = 10.0 * np.log10(pxx / np.square(p_ref))
    return fxx, pxx


def freq2erb(freq):
    """
    Convert frequency in Hz to ERB-number. Same as `freqtoerb.m` in the AMT.
    """
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb2freq(erb):
    """
    Convert ERB-number to frequency in Hz. Same as `erbtofreq.m` in the AMT.
    """
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(start, stop, num):
    """
    Create an array of frequencies in Hz evenly spaced on a ERB-number scale.
    Same as `erbspace.m` in the AMT.

    Args
    ----
    start (float): minimum frequency in Hz
    stop (float): maximum frequency Hz
    num (int): number of frequencies (length of array)

    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    """
    return erb2freq(np.linspace(freq2erb(start), freq2erb(stop), num=num))


def data_iso226_2003():
    """
    Return data from ISO226:2003 Table 1 (frequency
    in Hz and hearing thresholds in dB re 20e-6 Pa).
    """
    data = np.array(
        [
            [20.0, 78.5],
            [25.0, 68.7],
            [31.5, 59.5],
            [40.0, 51.1],
            [50.0, 44.0],
            [63.0, 37.5],
            [80.0, 31.5],
            [100.0, 26.5],
            [125.0, 22.1],
            [160.0, 17.9],
            [200.0, 14.4],
            [250.0, 11.4],
            [315.0, 8.6],
            [400.0, 6.2],
            [500.0, 4.4],
            [630.0, 3.0],
            [800.0, 2.2],
            [1000.0, 2.4],
            [1250.0, 3.5],
            [1600.0, 1.7],
            [2000.0, -1.3],
            [2500.0, -4.2],
            [3150.0, -6.0],
            [4000.0, -5.4],
            [5000.0, -1.5],
            [6300.0, 6.0],
            [8000.0, 12.6],
            [10000.0, 13.9],
            [12500.0, 12.3],
        ]
    )
    frequency_hz = data[:, 0]
    threshold_dbspl = data[:, 1]
    return frequency_hz, threshold_dbspl


def dbspl2dbhl(f, dbspl):
    """
    Converts vector of sound levels in dB SPL to dB HL by
    linearly interpolating ISO226:2003 hearing thresholds
    on a log frequency scale.
    """
    threshold_f, threshold_dbspl = data_iso226_2003()
    threshold_dbspl = np.interp(
        np.log(f),
        xp=np.log(threshold_f),
        fp=threshold_dbspl,
    )
    return dbspl - threshold_dbspl


def dbhl2dbspl(f, dbhl):
    """
    Converts vector of sound levels in dB HL to dB SPL by
    linearly interpolating ISO226:2003 hearing thresholds
    on a log frequency scale.
    """
    threshold_f, threshold_dbspl = data_iso226_2003()
    threshold_dbspl = np.interp(
        np.log(f),
        xp=np.log(threshold_f),
        fp=threshold_dbspl,
    )
    return dbhl + threshold_dbspl


class FIRFilterbank(torch.nn.Module):
    def __init__(self, fir, dtype=torch.float32, **kwargs_conv1d):
        """
        Finite impulse response (FIR) filterbank.

        Args
        ----
        fir (array_like): filter impulse response with shape
            [n_taps] or [n_filters, n_taps]
        dtype (torch.dtype): data type to cast `fir` to if `fir`
            is not a `torch.Tensor`
        kwargs_conv1d (kwargs): keyword arguments passed on to
            torch.nn.functional.conv1d (must not include `groups`,
            which is used for batching)
        """
        super().__init__()
        if not isinstance(fir, (list, np.ndarray, torch.Tensor)):
            raise TypeError(f"unrecognized {type(fir)=}")
        if isinstance(fir, (list, np.ndarray)):
            fir = torch.tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            raise ValueError(f"invalid {fir.shape=}")
        self.register_buffer("fir", fir)
        self.kwargs_conv1d = kwargs_conv1d

    def forward(self, x, batching=False):
        """
        Apply filterbank along the time axis (dim=-1) via convolution
        in the time domain (torch.nn.functional.conv1d).

        Args
        ----
        x (torch.Tensor): input signal
        batching (bool): if True, the input is assumed to have shape
            [..., n_filters, time] and each channel is filtered with
            its own filter

        Returns
        -------
        y (torch.Tensor): filtered signal
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


class Hilbert(torch.nn.Module):
    def __init__(self, dim=-1):
        """
        Compute the analytic signal, using the Hilbert transform
        (torch implementation of `scipy.signal.hilbert`).
        """
        super().__init__()
        self.dim = dim

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, x):
        """ """
        n = x.shape[self.dim]
        X = torch.fft.fft(x, n=n, dim=self.dim, norm=None)
        h = torch.ones(n, dtype=X.dtype, device=X.device)
        if n % 2 == 0:
            h[1 : n // 2] *= 2
            h[(n // 2) + 1 : n] *= 0
        else:
            h[1 : (n + 1) // 2] *= 2
            h[(n + 1) // 2 : n] *= 0
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


class Interpolate(torch.nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        """ """
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, x):
        """ """
        return torch.nn.functional.interpolate(
            input=x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )


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


class GradientClippedPower(torch.autograd.Function):
    """
    Custom autograd Function for power function with gradient
    clipping (used to limit gradients for power compression).
    """

    @staticmethod
    def forward(ctx, input, power, clip_value):
        ctx.save_for_backward(input)
        ctx.power = power
        ctx.clip_value = clip_value
        return torch.pow(input, power)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = ctx.power * torch.pow(input, ctx.power - 1)
        grad = torch.clamp(grad, min=None, max=ctx.clip_value)
        return grad_output * grad, None, None


class GradientStableSigmoid(torch.autograd.Function):
    """
    Custom autograd Function for sigmoid function with stable
    gradient (avoid NaN due to overflow in rate-level function).
    """

    @staticmethod
    def forward(ctx, x, k, x0):
        ctx.save_for_backward(x, k, x0)
        return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

    @staticmethod
    def backward(ctx, grad_output):
        x, k, x0 = ctx.saved_tensors
        grad = k * torch.exp(-k * (x - x0))
        grad = grad / (torch.exp(-k * (x - x0)) + 1.0) ** 2
        grad = torch.nan_to_num(grad, nan=0.0, posinf=None, neginf=None)
        return grad_output * grad, None, None
