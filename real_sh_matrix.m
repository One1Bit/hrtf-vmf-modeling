function Y = real_sh_matrix(L, azimuth, elevation)
% REAL spherical harmonics matrix
%
% Input:
%   L         - SH order
%   azimuth   - [N x 1] in radians
%   elevation - [N x 1] in radians
%
% Output:
%   Y         - [N x (L+1)^2]

    N = length(azimuth);

    % convert elevation to colatitude
    phi = azimuth;
    theta = pi/2 - elevation;

    Y = zeros(N, (L+1)^2);
    col = 1;

    x = cos(theta(:))';   % row vector for legendre

    for l = 0:L
        P = legendre(l, x, 'sch');   % size: (l+1) x N

        for m = -l:l
            if m < 0
                mm = abs(m);
                Nlm = sqrt((2*l+1)/(4*pi));
                Y(:,col) = sqrt(2) * Nlm * squeeze(P(mm+1,:))' .* sin(mm * phi);
            elseif m == 0
                Nlm = sqrt((2*l+1)/(4*pi));
                Y(:,col) = Nlm * squeeze(P(1,:))';
            else
                Nlm = sqrt((2*l+1)/(4*pi));
                Y(:,col) = sqrt(2) * Nlm * squeeze(P(m+1,:))' .* cos(m * phi);
            end
            col = col + 1;
        end
    end
end