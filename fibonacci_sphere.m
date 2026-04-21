% midpoint Fibonacci sphere sampling
function [dir_N3, az, el] = fibonacci_sphere(N)

    i = (0:N-1)';
    golden_angle = pi * (3 - sqrt(5));

    z = 1 - 2*(i + 0.5)/N;
    r = sqrt(max(0, 1 - z.^2));
    phi = golden_angle * i;

    x = r .* cos(phi);
    y = r .* sin(phi);

    dir_N3 = [x y z];

    az = atan2(y, x);
    el = asin(z);
end