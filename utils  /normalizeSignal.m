function sigOut = normalizeSignal(sigIn)
% normalize signals to zero mean and unit variance
sigOut = sigIn - mean(sigIn);
sigOut = sigOut/std(sigOut);
end
