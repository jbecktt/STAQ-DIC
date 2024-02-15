function [U,F,stepwithinwhile,HGlobal,c0] = funICGNQuadtreeConformal(U0,x0_r,y0_r,RR,TT,XX_r,YY_r,Df,ImgRef,ImgDef,ImgWeight,winsize,tol,ICGNmethod)
%FUNCTION [U,F,stepwithinwhile,HGlobal] = funICGN(U0,x0,y0,Df,ImgRef,ImgDef,winsize,tol,method)
% The Local ICGN subset solver (part II): ICGN iteration 
% (see part I: ./func/LocalICGN.m)
% ----------------------------------------------
%   INPUT: U0                   Initial guess of the displacement fields
%          x0,y0                FE mesh nodal coordinates
%          Df                   Image grayscale value gradients
%          ImgRef               Reference image
%          ImgDef               Deformed image
%          winsize              DIC parameter subset size  
%          ICGNmethod           ICGN iteration scheme: 'GaussNewton' -or- 'LevenbergMarquardt'
%          tol                  ICGN iteration stopping threshold
%
%   OUTPUT: U                   Disp vector: [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           F                   Deformation gradient tensor
%                               F = [F11_node1, F21_node1, F12_node1, F22_node1, ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%           stepwithinwhile     ICGN iteration step for convergence
%           HGlobal             Hessian matrix for each local subset
%
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================

%% Initialization
warning('off');
% DfCropWidth = Df.DfCropWidth; % JGB: commented just to see where it comes up
imgSize = Df.imgSize;
winsize0 = winsize;
pad = 3;

debug = 0;

%% ---------------------------
% Find local subset region
% x = [x0-winsize/2 ; x0+winsize/2 ; x0+winsize/2 ; x0-winsize/2];  % [coordinates(elements(j,:),1)];
% y = [y0-winsize/2 ; y0+winsize/2 ; y0+winsize/2 ; y0-winsize/2];  % [coordinates(elements(j,:),2)];

% ---------------------------
% Initialization: Get P0
P0 = [0 0 0 0 U0(1) U0(2)]';
P = P0;

% ---------------------------
% Find region for f
% [XX,YY] = ndgrid([x(1):1:x(3)],[y(1):1:y(3)]); % JGB: commented
filter = (XX_r >= x0_r-winsize/2) & (XX_r <= x0_r+winsize/2) & (YY_r >= y0_r-winsize/2) & (YY_r <= y0_r+winsize/2); % JGB: new (currently assuming theta discontinuity case is not an issue)
[fr,fc] = find(filter);
fi = sub2ind(size(filter), fr, fc);

%tempf = imgfNormalizedbc.eval(XX,YY); 
%DfDx = imgfNormalizedbc.eval_Dx(XX,YY);
%DfDy = imgfNormalizedbc.eval_Dy(XX,YY);

%%%%%%%%%%%% !!!Mask START %%%%%%%%%%%%%%
% tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]); % JGB: comment
% tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
% tempfImgMask = Df.ImgRefMask.*double(filter); % JGB: new
% tempf = ImgRef.*tempfImgMask; % JGB: new
% tempf(isnan(tempf)) = 0; % JGB: new
tempfImgMask = Df.ImgRefMask(filter); % JGB: new vector
if any(isnan(tempfImgMask))
    error("NaN values encountered in the subset region")
end
tempImgWeight = ImgWeight(filter);
% A0 = numel(tempfImgMask);
A0 = ceil(sum(tempImgWeight));
% filter_msk = find(tempfImgMask~=1);
% if ~isempty(find(istempfImgMask))
    % filter(filter_msk) = 0;
    % tempfImgMask(filter_msk) = 0; % 2/7/24 change JGB
% end
tempf = ImgRef(filter); % JGB: new vector

%%%%%%%%%%%% !!!Mask END %%%%%%%%%%%%%%
% DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth)); % JGB: comment
% DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth)); % JGB: comment
% DfDx = Df.DfDx.*double(filter); % JGB: new (potentially convert NaN to 0)
% DfDy = Df.DfDy.*double(filter); % JGB: new (potentially convert NaN to 0)
DfDx = Df.DfDx(filter(1:end-pad,:)); % JGB: new (potentially convert NaN to 0) vector
DfDy = Df.DfDy(filter(1:end-pad,:)); % JGB: new (potentially convert NaN to 0) vector

%% %%%%%%%% If there are >60% of the subset are painted with patterns %%%%%%%%%%%%
maxIterNum = 100;
%%%%%%%%%%%% !!!Mask START %%%%%%%%%%%%%%
% DfDxImgMaskIndCount = sum(double(1-logical(tempf(:)))); % JGB: commment
% DfDxImgMaskIndCount = sum(double(1-logical(tempf(filter)))); % JGB: add (will throw error is a NaN value is encountered)
DfDxImgMaskIndCount = sum(double(1-logical(tempf)).*tempImgWeight); % JGB: add (will throw error is a NaN value is encountered) vector ------------------------- needs fixed currently tempf won't have any nan or 0
% [DfDxImgMaskIndRow,~] = find(abs(tempf)<1e-10);
% DfDxImgMaskIndCount = length(DfDxImgMaskIndRow);
%%%%%%%%%%%% !!!Mask END %%%%%%%%%%%%%%
  
% if DfDxImgMaskIndCount < 0.4*numel(tempf) %0.4*(winsize+1)^2 % JGB: edit vector
if DfDxImgMaskIndCount < 0.4*A0 % JGB: add
     
    if DfDxImgMaskIndCount > 0 %.0*(winsize+1)^2 % For those subsets where are 0's in the image mask file % JGB: edit
        % winsize = 2*max(ceil(sqrt((winsize+1)^2/((winsize+1)^2-DfDxImgMaskIndCount))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels JGB: comment
        % winsize = 2*max(ceil(sqrt(numel(tempf(filter))/(numel(tempf(filter))-DfDxImgMaskIndCount))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels % JGB:add
        % winsize = 2*max(ceil(sqrt(numel(tempf)/(numel(tempf)-DfDxImgMaskIndCount))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels % JGB: new vector
        winsize = 2*max(ceil(sqrt(A0/(A0-DfDxImgMaskIndCount))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels % JGB: new vector
        % x = [x0-winsize/2 ; x0+winsize/2 ; x0+winsize/2 ; x0-winsize/2]; % Update x % JGB: comment
        % y = [y0-winsize/2 ; y0+winsize/2 ; y0+winsize/2 ; y0-winsize/2]; % Update y % JGB: comment
        % [XX,YY] = ndgrid([x(1):1:x(3)],[y(1):1:y(3)]); % JGB: comment
        % tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]); % JGB: comment
        % tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask; % JGB: comment
        % DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth)); % JGB: comment
        % DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth)); % JGB: comment
        filter = (XX_r >= x0_r-winsize/2) & (XX_r <= x0_r+winsize/2) & (YY_r >= y0_r-winsize/2) & (YY_r <= y0_r+winsize/2); % JGB: new
        [fr,fc] = find(filter);
        fi = sub2ind(size(filter), fr, fc);

        tempfImgMask = Df.ImgRefMask(filter); % JGB: new vector
        tempImgWeight = ImgWeight(filter);
        tempf = ImgRef(filter); % JGB: new vector
        DfDx = Df.DfDx(filter(1:end-pad,:)); % JGB: new (potentially convert NaN to 0) vector
        DfDy = Df.DfDy(filter(1:end-pad,:)); % JGB: new (potentially convert NaN to 0) vector
        % tempfImgMask = Df.ImgRefMask.*double(filter); % JGB: new
        % tempf = ImgRef.*tempfImgMask; % JGB: new
        % tempf(isnan(tempf)) = 0; % JGB: new
        % DfDx = Df.DfDx.*double(filter); % JGB: new (potentially convert NaN to 0)
        % DfDy = Df.DfDy.*double(filter); % JGB: new (potentially convert NaN to 0)
    end
    
    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
    %%%%% Find connected region to deal with possible continuities %%%%%
    % tempf_BW2 = bwselect(logical(tempfImgMask), floor((winsize+1)/2), floor((winsize+1)/2), 4 ); % JGB: comment
    % DfDx_BW2 = bwselect(logical(tempfImgMask), floor((winsize+1)/2), floor((winsize+1)/2), 4 ); % JGB: comment
    % DfDy_BW2 = bwselect(logical(tempfImgMask), floor((winsize+1)/2), floor((winsize+1)/2), 4 ); % JGB: comment
    rp = regionprops(filter,'Centroid'); % JGB:add ------ can be replaced by mean(fr),mean(fc)
    if debug; figure; imshow(filter); hold on; plot(rp.Centroid(1),rp.Centroid(2),'r*'); hold off; pause(1); end
    if size(rp.Centroid,1) > 1; disp("the subset is non-continuous"); error('subset is non-continuous'); end % JGB: add
    
    tempfImgMaskBW2 = filter; % JGB: add
    % tempfImgMaskBW2 = zeros(size(filter)); % JGB: add
    % tempfImgMaskBW2(fi) = tempfImgMask; % JGB: add
    % tempfImgMaskBW2(isnan(tempfImgMaskBW2)) = 0; % JGB: add
    tempf_BW2 = bwselect(logical(tempfImgMaskBW2), floor(rp.Centroid(1)), floor(rp.Centroid(2)), 4); % JGB: add
    DfDx_BW2 = bwselect(logical(tempfImgMaskBW2), floor(rp.Centroid(1)), floor(rp.Centroid(2)), 4); % JGB: add (unused in initial code and still so)
    DfDy_BW2 = bwselect(logical(tempfImgMaskBW2), floor(rp.Centroid(1)), floor(rp.Centroid(2)), 4); % JGB: add (unused in initial code and still so)
    tempf = tempf.*double(tempf_BW2(filter));
    tempImgWeight = tempImgWeight.*double(tempf_BW2(filter));
    DfDx = DfDx.*double(tempf_BW2(filter));
    DfDy = DfDy.*double(tempf_BW2(filter));
    % JGB: potentially make matrix smaller here (XX,YY,DfDx,DfDy)
    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    % XX = RR(filter); % JGB: add
    % YY = TT(filter); % JGB: add
    XX = fr.* double(tempf_BW2(filter)); % JGB: add
    YY = fc.* double(tempf_BW2(filter)); % JGB: add
    % x0_r = rp.Centroid(2)+1; % JGB: add
    % x0_r = interp1(1:size(TT,1),RR(:,1),rp.Centroid(2)); % JGB: add
    % y0_r = interp1(1:size(TT,2),TT(1,:),rp.Centroid(1)); % JGB: add
    x0 = rp.Centroid(2); % JGB: add
    y0 = rp.Centroid(1); % JGB: add

    H2 = zeros(6,6); DfDxSq = (DfDx.^2); DfDySq = (DfDy.^2); DfDxDfDy = DfDx.*DfDy;
    XXSq = (XX-x0).^2; YYSq = (YY-y0).^2; XXYY = (XX-x0).*(YY-y0); % JGB: edit
    % H2(1,1) = sum(sum(XXSq.*DfDxSq));       H2(1,2) = sum(sum(XXSq.*DfDxDfDy ));
    % H2(1,3) = sum(sum( XXYY.*DfDxSq ));     H2(1,4) = sum(sum( XXYY.*DfDxDfDy ));
    % H2(1,5) = sum(sum( (XX-x0).*DfDxSq ));  H2(1,6) = sum(sum( (XX-x0).*DfDxDfDy ));
    % H2(2,2) = sum(sum(XXSq.*DfDySq));       H2(2,3) = H2(1,4);
    % H2(2,4) = sum(sum( XXYY.*DfDySq ));     H2(2,5) = H2(1,6);
    % H2(2,6) = sum(sum( (XX-x0).*DfDySq ));  H2(3,3) = sum(sum( YYSq.*DfDxSq ));
    % H2(3,4) = sum(sum( YYSq.*DfDxDfDy ));   H2(3,5) = sum(sum( (YY-y0).*DfDxSq ));
    % H2(3,6) = sum(sum( (YY-y0).*DfDxDfDy ));H2(4,4) = sum(sum(YYSq.*DfDySq ));
    % H2(4,5) = H2(3,6);  H2(4,6) = sum(sum((YY-y0).*DfDySq)); H2(5,5) = sum(sum(DfDxSq));
    % H2(5,6) = sum(sum(DfDxDfDy)); H2(6,6) = sum(sum(DfDySq));
    H2(1,1) = sum(sum(XXSq.*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));       H2(1,2) = sum(sum(XXSq.*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));
    H2(1,3) = sum(sum( XXYY.*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));     H2(1,4) = sum(sum( XXYY.*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));
    H2(1,5) = sum(sum( (XX-x0).*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));  H2(1,6) = sum(sum( (XX-x0).*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));
    H2(2,2) = sum(sum(XXSq.*DfDySq.*tempImgWeight/max(tempImgWeight(:))));       H2(2,3) = H2(1,4);
    H2(2,4) = sum(sum( XXYY.*DfDySq.*tempImgWeight/max(tempImgWeight(:))));     H2(2,5) = H2(1,6);
    H2(2,6) = sum(sum( (XX-x0).*DfDySq.*tempImgWeight/max(tempImgWeight(:))));  H2(3,3) = sum(sum( YYSq.*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));
    H2(3,4) = sum(sum( YYSq.*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));   H2(3,5) = sum(sum( (YY-y0).*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));
    H2(3,6) = sum(sum( (YY-y0).*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));H2(4,4) = sum(sum(YYSq.*DfDySq.*tempImgWeight/max(tempImgWeight(:))));
    H2(4,5) = H2(3,6);  H2(4,6) = sum(sum((YY-y0).*DfDySq.*tempImgWeight/max(tempImgWeight(:)))); H2(5,5) = sum(sum(DfDxSq.*tempImgWeight/max(tempImgWeight(:))));
    H2(5,6) = sum(sum(DfDxDfDy.*tempImgWeight/max(tempImgWeight(:)))); H2(6,6) = sum(sum(DfDySq.*tempImgWeight/max(tempImgWeight(:))));
    H = H2 + H2' - diag(diag(H2));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%% Old codes: to compute H matrix by a for loop %%%%
    % tempCoordx = XX(:); tempCoordy = YY(:);
    % for tempij = 1:size(tempCoordx,1)
    %
    %         H = H + ([DfDx(tempCoordx(tempij)-DfCropWidth,tempCoordy(tempij)-DfCropWidth) DfDy(tempCoordx(tempij)-DfCropWidth,tempCoordy(tempij)-DfCropWidth)]*...
    %             [tempCoordx(tempij)-x0 0 tempCoordy(tempij)-y0 0 1 0; 0 tempCoordx(tempij)-x0 0 tempCoordy(tempij)-y0 0 1])'* ...
    %             ([DfDx(tempCoordx(tempij)-DfCropWidth,tempCoordy(tempij)-DfCropWidth) DfDy(tempCoordx(tempij)-DfCropWidth,tempCoordy(tempij)-DfCropWidth)]*...
    %             [tempCoordx(tempij)-x0 0 tempCoordy(tempij)-y0 0 1 0; 0 tempCoordx(tempij)-x0 0 tempCoordy(tempij)-y0 0 1]);
    %
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
    meanf = mean(tempf(abs(tempf)>1e-10)); % maybe unnecessary to filter given the fact that I filter out NaN (maybe should break when Nan?) and 
    bottomf = sqrt((length(tempf(abs(tempf)>1e-10))-1)*var(tempf(abs(tempf)>1e-10)));
    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    
    % --------------------------
    % Initialize while loop
    normOfWOld=2; normOfWNew=1; normOfWNewAbs=1; stepwithinwhile=0;
    switch ICGNmethod   % For Gauss-Newton method
        case 'LevenbergMarquardt'
            delta = 0.001; % For Levenberg-Marquardt method
            KappaOld=1e10; KappaNew=1e10; KappaStore=zeros(10,1); PStore=zeros(10,6);
        otherwise % 'GaussNewton'
            delta = 0;
    end
    
    while( (stepwithinwhile <= maxIterNum) && (normOfWNew>tol) && (normOfWNewAbs>tol) )
        
        stepwithinwhile = stepwithinwhile+1;
        
        if stepwithinwhile>1 && DfDxImgMaskIndCount>0
            % error("JGB: not fixed yet")
            % winsize = 2*max(ceil(sqrt((winsize0+1)^2/(sum(double(tempg_BW2(:)))))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels
            winsize = 2*max(ceil(sqrt(A0/sum(tempg_BW2(:)))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels % JGB: new vector
            % x = [x0-winsize/2 ; x0+winsize/2 ; x0+winsize/2 ; x0-winsize/2]; % Update x
            % y = [y0-winsize/2 ; y0+winsize/2 ; y0+winsize/2 ; y0-winsize/2]; % Update y
            % [XX,YY] = ndgrid([x(1):1:x(3)],[y(1):1:y(3)]);
            filter = (XX_r >= x0_r-winsize/2) & (XX_r <= x0_r+winsize/2) & (YY_r >= y0_r-winsize/2) & (YY_r <= y0_r+winsize/2); % JGB: new
            tempfImgMask = Df.ImgRefMask(filter); % JGB: new vector

            [fr,fc] = find(filter);
            fi = sub2ind(size(filter), fr, fc);
            % tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
            tempImgWeight = ImgWeight(filter);
            XX = fr.* double(tempfImgMask); % JGB: add
            YY = fc.* double(tempfImgMask); % JGB: add

            x0 = mean(XX);
            y0 = mean(YY);
        end
        
        % Find region for g
        % %[tempCoordy, tempCoordx] = meshgrid(y(1):y(3),x(1):x(3));
        % tempCoordxMat = XX - x0_r*ones(winsize+1,winsize+1);
        % tempCoordyMat = YY - y0_r*ones(winsize+1,winsize+1);
        % u22 = (1+P(1))*tempCoordxMat + P(3)*tempCoordyMat + (x0_r+P(5))*ones(winsize+1,winsize+1);
        % v22 = P(2)*tempCoordxMat + (1+P(4))*tempCoordyMat + (y0_r+P(6))*ones(winsize+1,winsize+1);
        tempCoordxMat = XX - x0*ones(size(XX,1),size(XX,2));
        tempCoordyMat = YY - y0*ones(size(XX,1),size(XX,2));
        u22 = (1+P(1))*tempCoordxMat + P(3)*tempCoordyMat + (x0+P(5))*ones(size(XX,1),size(XX,2)); 
        v22 = P(2)*tempCoordxMat + (1+P(4))*tempCoordyMat + (y0+P(6))*ones(size(XX,1),size(XX,2)); % OG
        % v22 = P(2)*tempCoordxMat + (1+P(4)).*tempCoordyMat + (y0+P(6))*ones(size(XX,1),size(XX,2)) + P(5)./XX; % 2/10/24 edit wild'in
        
        % row1 = find(u22<3); row2 = find(u22>imgSize(1)-2); row3 = find(v22<3); row4 = find(v22>imgSize(2)-2); % JGB: comment
        % row1 = find(u22(filter)<RR(1,1)); row2 = find(u22(filter)>=RR(end-2,1)); row3 = find(v22(filter)<3); row4 = find(v22(filter)>imgSize(2)-2); % JGB: add still needs fixed to deal with discontinuities
        % if ~isempty([row1; row2; row3; row4]) % JGB: comment
            % normOfWNew = 1e6; % warning('Out of image boundary!') % JGB: comment
            % break; % JGB: comment
        % else % JGB: comment
        if true 
            if debug; disp("DANGER: NaN critical safeguards have been turned off boiiiiiiiiiiiiiiis... buckle ye belt"); end
            
            %tempg = imggNormalizedbc.eval(u22,v22)
            tempg = ba_interp2(ImgDef, v22, u22, 'cubic'); % (c,r)
            
            % DgDxImgMaskIndCount = sum(double(1-logical(tempg(:))));
            DgDxImgMaskIndCount = sum(double(1-logical(tempg(:))).*tempImgWeight);
            
            
            %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
            if DfDxImgMaskIndCount>0.0*(winsize0+1)^2 || DgDxImgMaskIndCount>0.0*(winsize0+1)^2
                  
                %%%%% Find connected region to deal with possible continuities %%%%%
                % tempg_BW2 = logical(tempg); % This was alrady commented
                % tempg_BW2 = bwselect(logical(tempg), floor((winsize+1)/2), floor((winsize+1)/2), 8 ); % JGB: comment
                filter2L = zeros(range(fr)+1,range(fc)+1);
                fil2L = sub2ind(size(filter2L),fr-min(fr)+1,fc-min(fc)+1);
                filter2L(fil2L) = logical(tempg);
                tempg_BW2 = bwselect(filter2L, floor(rp.Centroid(1)-min(fc)+1), floor(rp.Centroid(2)-min(fr)+1), 8); % JGB: add
                
                [rowtemp,~] = find(tempg_BW2==0);
                if isempty(rowtemp)
                    error("JGB: this has not been fixed yet")
                    tempg_BW2 = tempfImgMask;
                    tempg_BW2 = bwselect(tempg_BW2, floor((winsize+1)/2), floor((winsize+1)/2), 8);
                end
                tempg = tempg .* double(tempg_BW2(fil2L));

                % tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask; % JGB: this is the line I'm working on right now 1/29/24
                tempf = ImgRef(filter) .* tempfImgMask; % JGB: this is the line I'm working on right now 1/29/24
                % filter2 = zeros(size(ImgRef));
                % fil2 = sub2ind(size(filter2),fr,fc);
                % filter2(fil2) = 
                tempf = tempf .* double(tempg_BW2(fil2L));

                % DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
                % DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
                DfDx = Df.DfDx(filter(1:end-pad,:)); % JGB: new (potentially convert NaN to 0) vector
                DfDy = Df.DfDy(filter(1:end-pad,:)); % JGB: new (potentially convert NaN to 0) vector
                
                % DfDx = DfDx .* double(tempg_BW2);
                % DfDy = DfDy .* double(tempg_BW2);
                DfDx = DfDx .* double(tempg_BW2(fil2L));
                DfDy = DfDy .* double(tempg_BW2(fil2L));
                 
                H2 = zeros(6,6); DfDxSq = (DfDx.^2); DfDySq = (DfDy.^2); DfDxDfDy = DfDx.*DfDy;
                XXSq = (XX-x0).^2; YYSq = (YY-y0).^2; XXYY = (XX-x0).*(YY-y0);
                % H2(1,1) = sum(sum(XXSq.*DfDxSq));       H2(1,2) = sum(sum(XXSq.*DfDxDfDy ));
                % H2(1,3) = sum(sum( XXYY.*DfDxSq ));     H2(1,4) = sum(sum( XXYY.*DfDxDfDy ));
                % H2(1,5) = sum(sum( (XX-x0).*DfDxSq ));  H2(1,6) = sum(sum( (XX-x0).*DfDxDfDy ));
                % H2(2,2) = sum(sum(XXSq.*DfDySq));       H2(2,3) = H2(1,4);
                % H2(2,4) = sum(sum( XXYY.*DfDySq ));     H2(2,5) = H2(1,6);
                % H2(2,6) = sum(sum( (XX-x0).*DfDySq ));  H2(3,3) = sum(sum( YYSq.*DfDxSq ));
                % H2(3,4) = sum(sum( YYSq.*DfDxDfDy ));   H2(3,5) = sum(sum( (YY-y0).*DfDxSq ));
                % H2(3,6) = sum(sum( (YY-y0).*DfDxDfDy ));H2(4,4) = sum(sum( YYSq.*DfDySq ));
                % H2(4,5) = H2(3,6);  H2(4,6) = sum(sum((YY-y0).*DfDySq)); H2(5,5) = sum(sum(DfDxSq));
                % H2(5,6) = sum(sum(DfDxDfDy)); H2(6,6) = sum(sum(DfDySq));
                H2(1,1) = sum(sum(XXSq.*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));       H2(1,2) = sum(sum(XXSq.*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));
                H2(1,3) = sum(sum( XXYY.*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));     H2(1,4) = sum(sum( XXYY.*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));
                H2(1,5) = sum(sum( (XX-x0).*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));  H2(1,6) = sum(sum( (XX-x0).*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));
                H2(2,2) = sum(sum(XXSq.*DfDySq.*tempImgWeight/max(tempImgWeight(:))));       H2(2,3) = H2(1,4);
                H2(2,4) = sum(sum( XXYY.*DfDySq.*tempImgWeight/max(tempImgWeight(:))));     H2(2,5) = H2(1,6);
                H2(2,6) = sum(sum( (XX-x0).*DfDySq.*tempImgWeight/max(tempImgWeight(:))));  H2(3,3) = sum(sum( YYSq.*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));
                H2(3,4) = sum(sum( YYSq.*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));   H2(3,5) = sum(sum( (YY-y0).*DfDxSq.*tempImgWeight/max(tempImgWeight(:))));
                H2(3,6) = sum(sum( (YY-y0).*DfDxDfDy.*tempImgWeight/max(tempImgWeight(:))));H2(4,4) = sum(sum( YYSq.*DfDySq.*tempImgWeight/max(tempImgWeight(:))));
                H2(4,5) = H2(3,6);  H2(4,6) = sum(sum((YY-y0).*DfDySq.*tempImgWeight/max(tempImgWeight(:)))); H2(5,5) = sum(sum(DfDxSq.*tempImgWeight/max(tempImgWeight(:))));
                H2(5,6) = sum(sum(DfDxDfDy.*tempImgWeight/max(tempImgWeight(:)))); H2(6,6) = sum(sum(DfDySq.*tempImgWeight/max(tempImgWeight(:))));
                H = H2 + H2' - diag(diag(H2));

                %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
                meanf = mean(tempf(abs(tempf)>1e-10));
                bottomf = sqrt((length(tempf(abs(tempf)>1e-10))-1)*var(tempf(abs(tempf)>1e-10)));
                %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%

%                 figure(1); clf;
%                 subplot(3,2,1), surf(DfDx,'edgecolor','none'); title('DfDx');view(2); axis equal; axis tight;
%                 subplot(3,2,2), surf(DfDy,'edgecolor','none'); title('DfDy');view(2); axis equal; axis tight;
%                 subplot(3,2,3), surf(tempf0,'edgecolor','none'); title('tempf'); view(2); axis equal; axis tight;caxis([-1,1]);
%                 subplot(3,2,4), surf(tempg0,'edgecolor','none'); title('tempg'); view(2); axis equal; axis tight; caxis([-1,1]);
%                 subplot(3,2,5), imshow(flipud( tempfImgMask) ); title('im f mask');
%                 subplot(3,2,6), imshow(flipud( tempg_BW2) ); title('im g mask');
% 
%                 pause;

            end
            %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    
    
%              figure(1); clf;
%                 subplot(3,2,1), surf(DfDx,'edgecolor','none'); title('DfDx');view(2); axis equal; axis tight;
%                 subplot(3,2,2), surf(DfDy,'edgecolor','none'); title('DfDy');view(2); axis equal; axis tight;
%                 subplot(3,2,3), surf(tempf,'edgecolor','none'); title('tempf'); view(2); axis equal; axis tight;caxis([-1,1]);
%                 subplot(3,2,4), surf(tempg,'edgecolor','none'); title('tempg'); view(2); axis equal; axis tight; caxis([-1,1]);
%                 subplot(3,2,5), imshow(flipud( tempfImgMask) ); title('im f mask');
%               
% 
%                 pause;
                
                
            % ====== Old version codes ======
            % tempg = zeros(size(tempf,1)*size(tempf,2),1);
            % [tempCoordy, tempCoordx] = meshgrid(1:winsize+1,1:winsize+1);
            % tempCoordx = tempCoordx(:); tempCoordy = tempCoordy(:);
            %
            % for tempij = 1:size(tempCoordx,1)
            %     tempg(tempij)= ...
            %         fungInterpolation_g(u22(tempCoordx(tempij),tempCoordy(tempij)), v22(tempCoordx(tempij),tempCoordy(tempij)), ...
            %         g(floor(u22(tempCoordx(tempij),tempCoordy(tempij)))-1:floor(u22(tempCoordx(tempij),tempCoordy(tempij)))+2, ...
            %         floor(v22(tempCoordx(tempij),tempCoordy(tempij)))-1:floor(v22(tempCoordx(tempij),tempCoordy(tempij)))+2));
            % end
            %
            % tempg = reshape(tempg, winsize+1, winsize+1);
            % ===============================
            
            % A = [1+P(1) P(2) 0; P(3) 1+P(4) 0; P(5) P(6) 1];
            % tform = affine2d((A));
            %
            % tempg2 = g((x(1)-winsize/2):(x(3)+winsize/2), (y(1)-winsize/2):(y(3)+winsize/2));
            % tempg3 = imwarp(tempg2,tform,'cubic');
            %
            % figure; imshow(tempf,[]);
            % figure; imshow(tempg2,[]);
            % figure; imshow(tempg3,[]);
            %
            % [M,N] = size(tempg3)
            % tempg = tempg3(ceil((M+1)/2)-winsize/2:ceil((M+1)/2)+winsize/2, ceil((N+1)/2)-winsize/2:ceil((N+1)/2)+winsize/2);
            % figure; imshow(tempg,[]);
            
            %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
            %%%%% Find connected region to deal with possible continuities %%%%%
            meang = mean(tempg(abs(tempg)>1e-10));
            bottomg = sqrt((length(tempg(abs(tempg)>1e-10))-1)*var(tempg(abs(tempg)>1e-10)));
            %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
            
            % ============ For Levenberg-Marquardt method ============
            switch ICGNmethod
                case 'LevenbergMarquardt'
                    error("JGB: I have not done this part yet")
                    % Compute functinoal error
                    KappaOld = KappaNew;
                    Kappatemp = (tempf-meanf)/bottomf - (tempg-meang)/bottomg;
                    Kappatemp = Kappatemp.*Kappatemp;
                    KappaNew = sum(Kappatemp(:));
                    
                    if KappaNew < 1.02*KappaOld
                        delta = delta/10;
                    else
                        delta = delta*10;
                        % Perform P inverse
                        DeltaP = -DeltaP;
                        tempP1 =  (-DeltaP(1)-DeltaP(1)*DeltaP(4)+DeltaP(2)*DeltaP(3))/temp;
                        tempP2 =  -DeltaP(2)/temp;
                        tempP3 =  -DeltaP(3)/temp;
                        tempP4 =  (-DeltaP(4)-DeltaP(1)*DeltaP(4)+DeltaP(2)*DeltaP(3))/temp;
                        tempP5 =  (-DeltaP(5)-DeltaP(4)*DeltaP(5)+DeltaP(3)*DeltaP(6))/temp;
                        tempP6 =  (-DeltaP(6)-DeltaP(1)*DeltaP(6)+DeltaP(2)*DeltaP(5))/temp;
                        
                        tempMatrix = [1+P(1) P(3) P(5); P(2) 1+P(4) P(6); 0 0 1]*...
                            [1+tempP1 tempP3 tempP5; tempP2 1+tempP4 tempP6; 0 0 1];
                        
                        P1 = tempMatrix(1,1)-1;
                        P2 = tempMatrix(2,1);
                        P3 = tempMatrix(1,2);
                        P4 = tempMatrix(2,2)-1;
                        P5 = tempMatrix(1,3);
                        P6 = tempMatrix(2,3);
                        P = [P1 P2 P3 P4 P5 P6]';
                    end
                    
                    % Find region for g
                    % [tempCoordy, tempCoordx] = meshgrid(y(1):y(3),x(1):x(3));
                    %Repeated! tempCoordx = XX - x0*ones(winsize+1,winsize+1);
                    %Repeated! tempCoordy = YY - y0*ones(winsize+1,winsize+1);
                    u22 = (1+P(1))*tempCoordxMat + P(3)*tempCoordyMat + (x0+P(5))*ones(winsize+1,winsize+1);
                    v22 = P(2)*tempCoordxMat + (1+P(4))*tempCoordyMat + (y0+P(6))*ones(winsize+1,winsize+1);
                    
                    tempg = ImgDef.eval(u22,v22);
                    
                    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
                    %%%%% Find connected region to deal with possible continuities %%%%%
                    tempg_BW2 = bwselect(logical(tempg), floor((winsize+1)/2), floor((winsize+1)/2), 4 );
                    tempg = tempg .* double(tempg_BW2);
                    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
            
                    % ====== Old version codes ======
                    % tempg = zeros(size(tempf,1)*size(tempf,2),1);
                    %
                    % [tempCoordy, tempCoordx] = meshgrid(1:winsize+1,1:winsize+1);
                    % tempCoordx = tempCoordx(:); tempCoordy = tempCoordy(:);
                    %
                    % parfor tempij = 1:size(tempCoordx,1)
                    %     tempg(tempij)= ...
                    %         fungInterpolation_g(u22(tempCoordx(tempij),tempCoordy(tempij)), v22(tempCoordx(tempij),tempCoordy(tempij)), ...
                    %         g(floor(u22(tempCoordx(tempij),tempCoordy(tempij)))-1:floor(u22(tempCoordx(tempij),tempCoordy(tempij)))+2, ...
                    %         floor(v22(tempCoordx(tempij),tempCoordy(tempij)))-1:floor(v22(tempCoordx(tempij),tempCoordy(tempij)))+2));
                    % end
                    %
                    % tempg = reshape(tempg, winsize+1, winsize+1);
                    % ==================================
                    
                    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
                    %%%%% Find connected region to deal with possible continuities %%%%%
                    meang = mean(tempg(abs(tempg)>1e-10));
                    bottomg = sqrt((length(tempg(abs(tempg)>1e-10))-1)*var(tempg(abs(tempg)>1e-10)));
                    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
                    
                otherwise
            end
            
            % % ============ End of Levenberg-Marquardt method ============
            
            % ====== Assemble b vector old version ======
            % b = zeros(6,1);
            %
            % %[tempCoordy, tempCoordx] = meshgrid(y(1):y(3),x(1):x(3));
            % %tempCoordx = tempCoordx(:); tempCoordy = tempCoordy(:);
            %
            % for tempij = 1:size(tempCoordx,1)
            %     b = b + bottomf*([DfDx(tempCoordx(tempij)-DfCropWidth,tempCoordy(tempij)-DfCropWidth) DfDy(tempCoordx(tempij)-DfCropWidth,tempCoordy(tempij)-DfCropWidth)]*...
            %             [tempCoordx(tempij)-x0 0 tempCoordy(tempij)-y0 0 1 0; 0 tempCoordx(tempij)-x0 0 tempCoordy(tempij)-y0 0 1])'* ...
            %             ((tempf(tempCoordx(tempij)+1-x(1), tempCoordy(tempij)+1-y(1))-meanf)/bottomf - ...
            %             (tempg(tempCoordx(tempij)+1-x(1), tempCoordy(tempij)+1-y(1))-meang)/bottomg);
            % end
            % ====== Assemble b vector fast version ======
            b2 = zeros(6,1);
            % tempfMinustempg = (tempf-meanf*ones(winsize+1,winsize+1))/bottomf - (tempg-meang*ones(winsize+1,winsize+1))/bottomg; % JGB: comment
            tempfMinustempg = tempImgWeight.*((tempf-meanf*ones(size(tempf)))/bottomf - (tempg-meang*ones(size(tempg)))/bottomg)/max(tempImgWeight(:));
            % tempfMinustempg = ((tempf-meanf*ones(size(tempf)))/bottomf - (tempg-meang*ones(size(tempg)))/bottomg);
            b2(1) = sum(sum( (XX-x0).*DfDx.*tempfMinustempg ));
            b2(2) = sum(sum( (XX-x0).*DfDy.*tempfMinustempg ));
            b2(3) = sum(sum( (YY-y0).*DfDx.*tempfMinustempg ));
            b2(4) = sum(sum( (YY-y0).*DfDy.*tempfMinustempg ));
            b2(5) = sum(sum( DfDx.*tempfMinustempg ));
            b2(6) = sum(sum( DfDy.*tempfMinustempg ));
            % b2(1) = sum(sum( (XX-x0).*DfDx.*tempfMinustempg.*tempImgWeight/max(tempImgWeight(:)) ));
            % b2(2) = sum(sum( (XX-x0).*DfDy.*tempfMinustempg.*tempImgWeight/max(tempImgWeight(:)) ));
            % b2(3) = sum(sum( (YY-y0).*DfDx.*tempfMinustempg.*tempImgWeight/max(tempImgWeight(:)) ));
            % b2(4) = sum(sum( (YY-y0).*DfDy.*tempfMinustempg.*tempImgWeight/max(tempImgWeight(:)) ));
            % b2(5) = sum(sum( DfDx.*tempfMinustempg.*tempImgWeight/max(tempImgWeight(:)) ));
            % b2(6) = sum(sum( DfDy.*tempfMinustempg.*tempImgWeight/max(tempImgWeight(:)) ));
            
            b = bottomf * b2;
            
            normOfWOld = normOfWNew;
            normOfWNew = norm(b(:)); normOfWNewAbs = normOfWNew;
            
            if stepwithinwhile ==1
                normOfWNewInit = normOfWNew;
            end
            if normOfWNewInit > tol
                normOfWNew = normOfWNew/normOfWNewInit;
            else
                normOfWNew = 0;
            end
            
            if (normOfWNew<tol) || (normOfWNewAbs<tol)
                break
            else
                % DeltaP = [0 0 0 0 0 0];
                % tempH = (H + delta*diag(diag(H)));
                % tempb = b;
                % DeltaP(5:6) = -tempH(5:6,5:6)\tempb(5:6);
                DeltaP = -(H + delta*diag(diag(H))) \ b;
                detDeltaP =  ((1+DeltaP(1))*(1+DeltaP(4)) - DeltaP(2)*DeltaP(3));
                if (detDeltaP ~= 0)
                    tempP1 =  (-DeltaP(1)-DeltaP(1)*DeltaP(4)+DeltaP(2)*DeltaP(3))/detDeltaP;
                    tempP2 =  -DeltaP(2)/detDeltaP;
                    tempP3 =  -DeltaP(3)/detDeltaP;
                    tempP4 =  (-DeltaP(4)-DeltaP(1)*DeltaP(4)+DeltaP(2)*DeltaP(3))/detDeltaP;
                    tempP5 =  (-DeltaP(5)-DeltaP(4)*DeltaP(5)+DeltaP(3)*DeltaP(6))/detDeltaP;
                    tempP6 =  (-DeltaP(6)-DeltaP(1)*DeltaP(6)+DeltaP(2)*DeltaP(5))/detDeltaP;
                    
                    tempMatrix = [1+P(1) P(3) P(5); P(2) 1+P(4) P(6); 0 0 1]*...
                        [1+tempP1 tempP3 tempP5; tempP2 1+tempP4 tempP6; 0 0 1];
                    
                    P1 = tempMatrix(1,1)-1;
                    P2 = tempMatrix(2,1);
                    P3 = tempMatrix(1,2);
                    P4 = tempMatrix(2,2)-1;
                    P5 = tempMatrix(1,3);
                    P6 = tempMatrix(2,3);
                    P = [P1 P2 P3 P4 P5 P6]';
                else
                    disp(['Det(DeltaP)==0!'])
                    break
                end
                
            end
        end
    end % end of while
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    if (normOfWNew<tol) || (normOfWNewAbs<tol)
        % elementsLocalMethodConvergeOrNot = 1;
    else
        stepwithinwhile = maxIterNum+1;
    end
    
    if (isnan(normOfWNew)==1)
        stepwithinwhile = maxIterNum+1;
    end
    if sum(abs(tempf(:))) < 1e-6
        stepwithinwhile = maxIterNum+3;
    end
        
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
else % if norm(diag(H)) > abs(eps)
    
    H = zeros(6,6);
    stepwithinwhile = maxIterNum+2;
    
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U(1) = P(5); U(2) = P(6);
F(1) = P(1); F(2) = P(2); F(3) = P(3); F(4) = P(4);

HGlobal = [H(1:6) H(8:12) H(15:18) H(22:24) H(29:30) H(36)];

c0(1) = sum(XX.*tempImgWeight/max(tempImgWeight))/sum(tempImgWeight/max(tempImgWeight));
c0(2) = sum(YY.*tempImgWeight/max(tempImgWeight))/sum(tempImgWeight/max(tempImgWeight));
% c0 = [interp2(XX_r,y0,x0),interp2(YY_r,y0,x0)];
c0 = [interp2(XX_r,c0(2),c0(1)),interp2(YY_r,c0(2),c0(1))];

if isnan(c0(1)) || isnan(c0(2))
    disp('f')
end

end
