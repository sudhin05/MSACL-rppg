function snr = get_SNR(pulseEst,gt_HR,fps)
mFreq = gt_HR/60;
harmFreq = 2*mFreq;
halfWinLength = 0.2; % Hz

N = 60*fps;
n = 2^nextpow2(N);
Y_pulse = fft(pulseEst,n);
powsPulse = abs(Y_pulse);
maxPow = max(powsPulse);
% powsPulse = powsPulse/maxPow;
freqPulse = (0:n/2)*fps/n;
% figure(22); clf; plot(freqPulse,powsPulse(1:length(freqPulse)));

freqRange = ((freqPulse <= mFreq+halfWinLength) & (freqPulse >= mFreq-halfWinLength))...
    | ((freqPulse <= harmFreq+halfWinLength) & (freqPulse >= harmFreq-halfWinLength));
powsPulse2 = powsPulse(1:length(freqPulse));
% figure(22); hold on; plot(freqPulse(freqRange),powsPulse2(freqRange));

freqRangeComp = (freqPulse <= 5) & (freqPulse >= 0.7) & (~freqRange);

snr = 10*log10(sum(powsPulse(freqRange).^2)/sum(powsPulse(freqRangeComp).^2));
end
