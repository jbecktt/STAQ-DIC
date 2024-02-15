function varargout = Cart2Rad(DataXY,x0,y0,xc,yc,string,pad)

if nargin == 6; pad = 0; end
if xc < min(x0(:)) || xc > max(x0(:)) || yc < min(y0(:)) || yc > max(y0(:))
    error('Centroid of transformation (xc,yc) must be within the limits of x0 and y0.')
end

% Calculate radial coordiantes
r0 = sqrt((x0-xc).^2+(y0-yc).^2);
theta0 = atan2d(y0-yc,x0-xc);
theta0(theta0 == -180) = 180; % wraping to (-180,180] from [-180,180] JGB: unnecessary?

% Calculate values for the interpolant with overlapped values to deal with the discontinuity at between 180 and -180°.
% (this is currently done in a slow manner -- can be cleaned-up for speed at a later time)
filter_high = theta0<=0;
filter_low = theta0>=0;

r0_high = r0(filter_high);
r0_low = r0(filter_low); 
theta0_high = theta0(filter_high) + 360;
theta0_low = theta0(filter_low) - 360; 
dataxy_high = DataXY(filter_high);
dataxy_low = DataXY(filter_low); 
r0_overlap = [reshape(r0,[],1); r0_high; r0_low];
theta0_overlap = [reshape(theta0,[],1); theta0_high; theta0_low];
dataxy_overlap = [reshape(DataXY,[],1);dataxy_high;dataxy_low];

% Calcuate boundary values for cropping
r_b = [r0(1,1:end), r0(2:end,end)', r0(end,1:end-1), r0(2:end-1,1)'];
theta_b= [theta0(1,1:end), theta0(2:end,end)', theta0(end,1:end-1), theta0(2:end-1,1)'];
[theta_b,I] = sort(theta_b); 
r_b = r_b(I);

% Space grid to ensure no data is lost
rmax = max(r0(:));
rmin = min(r0(:));
% [rmaxr,rmaxc] = ind2sub(size(r0),rmaxi);
% [rminr,rminc] = ind2sub(size(r0),rmini);
[rmaxr,rmaxc] = find(r0 == rmax); 
[rminr,rminc] = find(r0 == rmin);
theta0ofrmax = theta0(r0==rmax);

t0tempindex = linspace(-180,180,ceil(2*pi*rmax)); % r changes the quickest when close to 90°*x where x is an integer so this holds
t0tempindex = t0tempindex(2:end);
rn = max(abs([rmaxr-rminr,rmaxc-rminr,rmaxr-rminc,rmaxc-rminc]),[],'all');
r0tempindex = 0:1:ceil(rmax); % r closest spaced along cartesian axes 
% r0tempindex = linspace(0,ceil(rmax),rn); % r closest spaced along cartesian axes 
[t0temp,r0temp] = meshgrid(t0tempindex,r0tempindex);

if pad > 0
   r0temp = [flipud(r0temp(2:2+pad-1,:));r0temp]; % pad in -r direction (currently assumes that the first row is r = 0)
   t0temp = [wrapTo180(t0temp(2:2+pad-1,:)+180);t0temp];
   r0temp = [r0temp(:,end-pad+1:end),r0temp,r0temp(:,1:pad)]; % pad discontinuity in theta
   t0temp = [t0temp(:,end-pad+1:end),t0temp,t0temp(:,1:pad)]; 
end

if string == "cubic" % to be used for DIC images
    DataRT = griddata(r0_overlap,theta0_overlap,dataxy_overlap,r0temp,t0temp,"cubic"); % possibly replace with ba_interp2 to boost speed
    DataRT(isnan(DataRT)) = 0; 
elseif string == "mask" % to be used for mask images
    interp = scatteredInterpolant(r0_overlap,theta0_overlap,dataxy_overlap,"linear","linear"); % if changed to none it essentially will only effect the point(s) that equal rmax
    DataRT = interp(r0temp,t0temp);
    DataRT = double(logical(ceil(DataRT)));
    for i = 1:size(DataRT,2) % cropping most extrapolated values
        Ttemp = t0temp(1,i);
        Rmaxtemp = interp1(theta_b,r_b,Ttemp,"spline","extrap");
        filter = double(r0temp(:,i)<=ceil(Rmaxtemp));
        filter(filter == 0) = NaN;
        DataRT(:,i) = DataRT(:,i).*filter;
    end
    if any(~ismember(DataRT(:),[0 1]).*~isnan(DataRT(:))); error("Radial mask has values besides 0, 1 and NaN!!"); end
end

for i = 1:nargout
    if i == 1
        varargout{i} = DataRT;
    elseif i == 2
        varargout{i} = r0temp;
    elseif i == 3
        varargout{i} = t0temp;
    elseif i == 4 
        x0temp = r0temp.*cosd(t0temp);
        varargout{i} = x0temp;
    elseif i == 5 
        y0temp = r0temp.*sind(t0temp);
        varargout{i} = y0temp;
    elseif i ==6 
        weight = 0.5*((r0temp+0.5).^2-(r0temp-0.5).^2)*(deg2rad(t0temp(1,2))-deg2rad(t0temp(1,1)));
        varargout{i} = weight;
    end
end


% possibly export x and y equivalent for all of the points
% possibly switch over to ba_interp
% potentially make a more general function that could help to transform the meshes
% deal with overlapping
% error if xc and yc don't fall in range of x and y terms

% 2*pi*rmax is conservative
% this could probably be expanded to include outside of the circle stuff
