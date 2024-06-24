function Pearson_coeff = get_Pearson_coeff(X, Y)
% Pearson_correlation_coefficient
% input£ºX Y
% output£ºPearson_coeff
if length(X) ~= length(Y)
    error('Dimension is not equal');
end
x_m = mean(X);
y_m = mean(Y);
fenzi = sum((X-x_m).*(Y-y_m));
fenmu = sqrt(sum((X-x_m).^2)) .* sqrt(sum((Y-y_m).^2));
Pearson_coeff = fenzi / fenmu;
end

