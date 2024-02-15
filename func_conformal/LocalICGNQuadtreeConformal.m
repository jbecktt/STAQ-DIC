function [U,F,HtempPar,LocalTime,ConvItPerEle,LocalICGNBadPtNum,markCoordHoleStrain] = ...
                    LocalICGNQuadtreeConformal(U0,coordinatesFEM,RR180,TT180,XX180,YY180,Df180,ImgRef180,ImgDef180,ImgWeight180,DICpara,ICGNmethod,tol)
%FUNCTION [U,F,HtempPar,LocalTime,ConvItPerEle,LocalICGNBadPtNum] = LocalICGN(U0,coordinatesFEM,...
%                                       Df,imgfNormalizedbc,imggNormalizedbc,DICpara,ICGNmethod,tol)
% The Local ICGN subset solver (part I): to assign a sequential computing or a parallel computing
% (see part II: ./func/funICGN.m)
% ----------------------------------------------
%   INPUT: U0                   Initial guess of the displacement fields
%          coordinatesFEM       FE mesh coordinates
%          Df                   Image grayscale value gradients
%          ImgRef               Reference image
%          ImgDef               Deformed image
%          DICpara              DIC parameters: subset size, subset spacing, ...
%          ICGNmethod           ICGN iteration scheme: 'GaussNewton' -or- 'LevenbergMarquardt'
%          tol                  ICGN iteration stopping threshold
%
%   OUTPUT: U                   Disp vector: [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           F                   Deformation gradient tensor
%                               F = [F11_node1, F21_node1, F12_node1, F22_node1, ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%           HtempPar            Hessian matrix for each local subset
%           LocalTime           Computation time
%           ConvItPerEle        ICGN iteration step for convergence
%           LocalICGNBadPtNum   Number of subsets whose ICGN iterations don't converge 
% 
% ----------------------------------------------
% Reference
% [1] RegularizeNd. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/61436-regularizend
% [2] Gridfit. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit
% [3] Rbfinterp. Matlab File Exchange open source.
% https://www.mathworks.com/matlabcentral/fileexchange/10056-scattered-data-interpolation-and-approximation-using-radial-base-functions
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================
 

%% Initialization
warning('off');
winsize = DICpara.winsize;
winstepsize = DICpara.winstepsize;
ClusterNo = DICpara.ClusterNo;
debug = 0;

temp = zeros(size(coordinatesFEM,1),1); UtempPar = temp; VtempPar = temp; % UtempPar = UPar{1}; VtempPar = UPar{2};
F11tempPar = temp; F21tempPar = temp; F12tempPar = temp; F22tempPar = temp;% F11tempPar = FPar{1}; F21tempPar = FPar{2}; F12tempPar = FPar{3}; F22tempPar = FPar{4};
HtempPar = zeros(size(coordinatesFEM,1),21);
ConvItPerEle = zeros(size(coordinatesFEM,1),1);
markCoordHoleStrainOrNot = zeros(size(coordinatesFEM,1),1);
  
% -------- How to change parallel pools ---------
% myCluster = parcluster('local');
% myCluster.NumWorkers = 4;  % 'Modified' property now TRUE
% saveProfile(myCluster);    % 'local' profile now updated,
%                            % 'Modified' property now FALSE
% -------------- Or we can do this --------------
% Go to the Parallel menu, then select Manage Cluster Profiles.
% Select the "local" profile, and change NumWorkers to 4.
% -----------------------------------------------

%% ClusterNo == 0 or 1: Sequential computing
if (ClusterNo == 0) || (ClusterNo == 1)

    h = waitbar(0,'Please wait for Subproblem 1 IC-GN iterations!'); tic;

    [~,minTi360] = min(abs(wrapTo360(TT180(1,:))));
    XX360 = [XX180(:,minTi360:end),XX180(:,1:minTi360-1)]; YY360 = [YY180(:,minTi360:end),YY180(:,1:minTi360-1)];
    RR360 = [RR180(:,minTi360:end),RR180(:,1:minTi360-1)]; TT360 = [TT180(:,minTi360:end),TT180(:,1:minTi360-1)]; % JGB: technically RR360 vs RR180 and could just replaced with a single value
    Df360 = Df180; Df360.DfDx = [Df360.DfDx(:,minTi360:end),Df360.DfDx(:,1:minTi360-1)]; Df360.DfDy = [Df360.DfDy(:,minTi360:end),Df360.DfDy(:,1:minTi360-1)];
    Df360.ImgRefMask = [Df360.ImgRefMask(:,minTi360:end),Df360.ImgRefMask(:,1:minTi360-1)]; ImgWeight360 = [ImgWeight180(:,minTi360:end),ImgWeight180(:,1:minTi360-1)];
    ImgRef360 = [ImgRef180(:,minTi360:end),ImgRef180(:,1:minTi360-1)]; ImgDef360 = [ImgDef180(:,minTi360:end),ImgDef180(:,1:minTi360-1)];


    for tempj = 1 : size(coordinatesFEM,1)  % tempj is the element index
        
        x0temp = round(coordinatesFEM(tempj,1)); y0temp = round(coordinatesFEM(tempj,2));  
        
        try
            % x = [x0temp-winsize/2 ; x0temp+winsize/2 ; x0temp+winsize/2 ; x0temp-winsize/2];  % [coordinates(elements(j,:),1)];
            % y = [y0temp-winsize/2 ; y0temp+winsize/2 ; y0temp+winsize/2 ; y0temp-winsize/2];  % [coordinates(elements(j,:),2)];
            % tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
            % tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
            XX = XX180; YY = YY180; RR = RR180; TT = TT180; ImgRef = ImgRef180; ImgDef = ImgDef180; Df = Df180; ImgWeight = ImgWeight180;
            filter = (XX >= x0temp-winsize/2) & (XX <= x0temp+winsize/2) & (YY >= y0temp-winsize/2) & (YY <= y0temp+winsize/2);
            rp = regionprops(filter,'Centroid');
            if length(rp) == 2 || length(rp) == 1 && (abs(rp.Centroid(1) - size(filter,2)/2) >  size(filter,2)/4)
                filter = [filter(:,minTi360:end),filter(:,1:minTi360-1)];
                rp = regionprops(filter,'Centroid');
                if length(rp) == 1
                    XX = XX360; YY = YY360; RR = RR360; TT = TT360; ImgRef = ImgRef360; ImgDef = ImgDef360; Df = Df360; ImgWeight = ImgWeight360;
                else
                    ConvItPerEle(tempj) = 104;
                    error("Unable to create continuous subset by shifting theta discontinuity.")
                end
            elseif length(rp) > 2 % JGB: fix this later... I need to make this more restrictive... check if pixels are in final columns
                ConvItPerEle(tempj) = 105;
                error("Greater than 3 seperate regions exist in the image")
            end
            tempfImgMask = Df.ImgRefMask.*filter;
            tempf = ImgRef.*tempfImgMask;
            tempImgWeight = ImgWeight.*tempfImgMask;
            tempfcount = tempf(filter); % make an alternative variable to count masked regions that does not include NaN values
            tempfcount(isnan(tempfcount)) = 0;
            DfDxImgMaskIndCount = sum(double(1-logical(tempfcount(:))));
            if DfDxImgMaskIndCount > 0
                markCoordHoleStrainOrNot(tempj) = 1;
            end
        catch
            markCoordHoleStrainOrNot(tempj) = 1;
        end
        
        try 
            [Utemp, Ftemp, ConvItPerEle(tempj), HtempPar(tempj,:),c0(tempj,:)] = funICGNQuadtreeConformal(U0(2*tempj-1:2*tempj), ...
                               x0temp,y0temp,RR,TT,XX,YY,Df,ImgRef,ImgDef,ImgWeight,winsize,tol,ICGNmethod);
            % [Utemp, Ftemp, ConvItPerEle(tempj), HtempPar(tempj,:)] = funICGNQuadtree(U0(2*tempj-1:2*tempj), ...
                               % x0temp,y0temp,Df,ImgRef,ImgDef,winsize,tol,ICGNmethod);
           % disp(['ele ',num2str(tempj),' converge step is ',num2str(ConvItPerEle(tempj)),' (>0-converged; 0-unconverged)']);
            % ------ Store solved deformation gradients ------
            UtempPar(tempj) = Utemp(1); VtempPar(tempj) = Utemp(2); 
            F11tempPar(tempj) = Ftemp(1); F21tempPar(tempj) = Ftemp(2); F12tempPar(tempj) = Ftemp(3); F22tempPar(tempj) = Ftemp(4);
           % waitbar(tempj/(size(coordinatesFEM,1)));
        catch
           ConvItPerEle(tempj) = -1;
           UtempPar(tempj) = nan; VtempPar(tempj) = nan;
           F11tempPar(tempj) = nan; F21tempPar(tempj) = nan; F12tempPar(tempj) = nan; F22tempPar(tempj) = nan;
           waitbar(tempj/(size(coordinatesFEM,1)));
        end
        if debug
            disp("Subset number:"+num2str(tempj)+" ConvItPerEle:"+num2str(ConvItPerEle(tempj)))
            close all
        end
        % pause(3)
    end
    close(h); LocalTime = toc;
    
%% ClusterNo > 1: parallel computing
else
    error('Parallel computing no supported for conformal subsets yet! - JGB')
    
    % Start parallel computing
    % ****** This step needs to be careful: may be out of memory ******
    disp('--- Set up Parallel pool ---'); tic;
    hbar = parfor_progressbar(size(coordinatesFEM,1),'Please wait for Subproblem 1 IC-GN iterations!');
    parfor tempj = 1:size(coordinatesFEM,1)  % tempj is the element index
        
        x0temp = round(coordinatesFEM(tempj,1)); y0temp = round(coordinatesFEM(tempj,2));  
         
        try
            x = [x0temp-winsize/2 ; x0temp+winsize/2 ; x0temp+winsize/2 ; x0temp-winsize/2];  % [coordinates(elements(j,:),1)];
            y = [y0temp-winsize/2 ; y0temp+winsize/2 ; y0temp+winsize/2 ; y0temp-winsize/2];  % [coordinates(elements(j,:),2)];
            tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
            tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
            DfDxImgMaskIndCount = sum(double(1-logical(tempf(:))));
            if DfDxImgMaskIndCount > 0. *(winsize+1)^2
                markCoordHoleStrainOrNot(tempj) = 1;
            end
        catch
            markCoordHoleStrainOrNot(tempj) = 1;
        end
        
        try 
            [Utemp, Ftemp, ConvItPerEle(tempj), HtempPar(tempj,:)] = funICGNQuadtree(U0(2*tempj-1:2*tempj), ...
                    x0temp,y0temp,Df,ImgRef,ImgDef,winsize,tol,ICGNmethod);
            % disp(['ele ',num2str(tempj),' converge step is ',num2str(ConvItPerEle(tempj)),' (>0-converged; 0-unconverged)']);
            % ------ Store solved deformation gradients ------
            UtempPar(tempj) = Utemp(1); VtempPar(tempj) = Utemp(2); 
            F11tempPar(tempj) = Ftemp(1); F21tempPar(tempj) = Ftemp(2); F12tempPar(tempj) = Ftemp(3); F22tempPar(tempj) = Ftemp(4);
            hbar.iterate(1);
        catch
            ConvItPerEle(tempj) = -1;
            UtempPar(tempj) = nan; VtempPar(tempj) = nan;
            F11tempPar(tempj) = nan; F21tempPar(tempj) = nan; F12tempPar(tempj) = nan; F22tempPar(tempj) = nan;
            hbar.iterate(1); 
        end
    end
     
    close(hbar); 
    LocalTime = toc;
    
end


figure, plot(coordinatesFEM(ConvItPerEle > 0 & ConvItPerEle <= 100,1),coordinatesFEM(ConvItPerEle > 0 & ConvItPerEle <= 100,2),'b.','MarkerSize', 20)
hold on
% plot(coordinatesFEM(ConvItPerEle == -1,1),coordinatesFEM(ConvItPerEle == -1,2),'g.','MarkerSize', 20) 
plot(coordinatesFEM(ConvItPerEle == 101,1),coordinatesFEM(ConvItPerEle == 101,2),'k.','MarkerSize', 20)
plot(coordinatesFEM(ConvItPerEle == 102,1),coordinatesFEM(ConvItPerEle == 102,2),'r.','MarkerSize', 20)
plot(coordinatesFEM(ConvItPerEle == 103,1),coordinatesFEM(ConvItPerEle == 103,2),'c.','MarkerSize', 20)
hold off
legend('good','101','102','103','Location','eastoutside','FontSize', 12)
axis equal; set(gca, 'YDir', 'reverse');

U = U0; U(1:2:end) = UtempPar; U(2:2:end) = VtempPar;
F = zeros(4*size(coordinatesFEM,1),1); F(1:4:end) = F11tempPar; F(2:4:end) = F21tempPar; F(3:4:end) = F12tempPar; F(4:4:end) = F22tempPar; 
markCoordHoleStrain = find(markCoordHoleStrainOrNot==1);

filt_conv = ConvItPerEle > 0 & ConvItPerEle < 100;
Urw = U(1:2:end).*(RR(2,1)-RR(1,1)); Utw= U(2:2:end).*(TT(1,2)-TT(1,1));
X = coordinatesFEM(:,1); Y = coordinatesFEM(:,2);
X = c0(:,1); Y = c0(:,2);
R = sqrt((X-200).^2 + (Y-200).^2); T = atan2d(Y-200,X-200);
Uxw = (R+Urw').*cosd((T+Utw')) - (X - 200); 
Uyw = (((R+Urw').*sind((T+Utw')) - (Y- 200)));

% defCase.R0_4 = 10;
% defCase.e_4 = 1.5; 
% defCase.N_4 = 8; 
% defCase.a_4 = 0.1;
% u1 = @(r0,theta0,t) 0.2.*((r0.^3 + ((1+defCase.e_4*t)*defCase.R0_4)^3 - defCase.R0_4^3).^(1/3) - r0).*(1 + defCase.a_4*sind(defCase.N_4*theta0));

u1 = @(r0,theta0,t) (2+1*cosd(2*theta0));


% save("test_123454321.m","Uxw","Uyw","Urw","Utw");
% Ux02 = (R+Ur0).*cosd((T+Ut0)) - (X - 200); 
% Uy02 = (R+Ur0).*sind((T+Ut0)) - (Y - 200); 

filt_conv = logical((R>0).*filt_conv.*filt_conv_OG123.*filt_conv_W123);
% Xtest = R.*cosd(T)+200; Ytest = R.*sind(T)+200;
figure;
tiledlayout(2,2)
nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Uxw(filt_conv),"filled"); title("Ux (px in OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; %clim([-0.03 0.03])
nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Uyw(filt_conv),"filled"); title("Uy (px in OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; %clim([-0.03 0.03])
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Uxw(filt_conv)-Ux0(filt_conv),"filled"); title("Uxw-Ux0 (px in OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; clim([-0.05 0.05])
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Uyw(filt_conv)-Uy0(filt_conv),"filled"); title("Uyw-Uy0 (px in OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; clim([-0.03 0.03])
nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Urw(filt_conv)'-u1(R(filt_conv),T(filt_conv),1),"filled"); title("Ur (px in  OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; %clim([-0.15 0.15])
nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Utw(filt_conv)',"filled"); title("Utheta (degrees -- CW+ as plotted)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; %clim([-0.1 0.1])
norm(Urw(filt_conv)'-u1(R(filt_conv),T(filt_conv),1))
norm(Utw(filt_conv)')
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Urw(filt_conv)'-Ur0(filt_conv),"filled"); title("Urw-Ur0 (px in OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; clim([-0.03 0.03])
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Utw(filt_conv)'-Ut0(filt_conv),"filled"); title("Uthetaw-Utheta0 (degrees -- CW+ as plotted)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; clim([-0.01 0.01]) 
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Urw(filt_conv)-Ur0(filt_conv)',"filled"); title("Urw-Ur0 (px in warped image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; 
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Utw(filt_conv)-Ut0(filt_conv)',"filled"); title("Utheta (degrees"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; 
% nexttile; histogram(Urw(filt_conv)-Ur0(filt_conv)');title("Urw-Ur0 (px in warped image)");
% nexttile; histogram(Utw(filt_conv)-Ut0(filt_conv)');title("Utw-Ut0 (degrees)");
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Uxw(filt_conv)-Ux0(filt_conv),"filled"); title("Uxw-Uw0 (px in OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; 
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Uyw(filt_conv)-Uy0(filt_conv),"filled"); title("Uyw-Uy0 (px in OG image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; 
% nexttile; histogram(Uxw(filt_conv)-Ux0(filt_conv));title("Uxw-Ux0 (px in warped image)");
% nexttile; histogram(Uyw(filt_conv)-Uy0(filt_conv));title("Uyw-Uy0 (degrees)");
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Xtest(filt_conv),"filled"); title("Ur (px in warped image)"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; 
% nexttile; scatter(coordinatesFEM(filt_conv,1),coordinatesFEM(filt_conv,2),[],Ytest(filt_conv),"filled"); title("Utheta (degrees"); axis equal; set(gca, 'YDir', 'reverse'); colorbar; 

ppr = sqrt(coordinatesFEM(:,1).^2 + coordinatesFEM(:,2).^2);
ppt = atan2d(coordinatesFEM(:,2),coordinatesFEM(:,1));

% ------ Clear bad points for Local DIC ------
% find bad points after Local Subset ICGN
maxIterNum = 100;
[row1,~] = find(ConvItPerEle(:)<0); 
[row2,~] = find(ConvItPerEle(:)>maxIterNum-1);
[row3,~] = find(ConvItPerEle(:)==maxIterNum+2);
LocalICGNBadPt = unique(union(row1,row2)); LocalICGNBadPtNum = length(LocalICGNBadPt)-length(row3);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Though some subsets are converged, but their accuracy is worse than most
% other subsets. This step is to remove those subsets with abnormal convergence steps
LocalICGNGoodPt = setdiff([1:1:size(coordinatesFEM,1)],LocalICGNBadPt);
%ConvItPerEleMean = median(ConvItPerEle(LocalICGNGoodPt));
%ConvItPerEleStd = std(ConvItPerEle(LocalICGNGoodPt));
pd = fitdist( ConvItPerEle(LocalICGNGoodPt), 'Normal'  );
ConvItPerEleMean = pd.mu;
ConvItPerEleStd = pd.sigma;
[row4,~] =  find(ConvItPerEle(:) > max([ConvItPerEleMean+1*ConvItPerEleStd, 6 ])); % Here "0.15" is an empirical value
LocalICGNBadPt = unique(union(LocalICGNBadPt,row4));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Print results info on the MATLAB command window
disp(['Local ICGN bad subsets %: ', num2str(LocalICGNBadPtNum),'/',num2str(size(coordinatesFEM,1)-length(row3)), ...
    '=',num2str(100*(LocalICGNBadPtNum)/(size(coordinatesFEM,1)-length(row3))),'%']);
U(2*LocalICGNBadPt-1) = NaN; U(2*LocalICGNBadPt) = NaN; 
F(4*LocalICGNBadPt-3) = NaN; F(4*LocalICGNBadPt-2) = NaN; F(4*LocalICGNBadPt-1) = NaN; F(4*LocalICGNBadPt) = NaN;
 
% figure,plot3(coordinatesFEM(:,1),coordinatesFEM(:,2),F(1:4:end),'.')
    

%%%%%% Fill nans %%%%%%
nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);
nanindexF = find(isnan(F(1:4:end))==1); notnanindexF = setdiff([1:1:size(coordinatesFEM,1)],nanindexF);


% dilatedI = ( imgaussfilt(double(Df.ImgRefMask),0.5) );
% dilatedI = logical( dilatedI > 0.5); % figure, imshow(dilatedI)
dilatedI = Df.ImgRefMask; 
cc = bwconncomp(dilatedI,8);
indPxAll = sub2ind( Df.imgSize, round(coordinatesFEM(:,1)), round(coordinatesFEM(:,2)) );
indPxNotNanAll = sub2ind( Df.imgSize, round(coordinatesFEM(notnanindex,1)), round(coordinatesFEM(notnanindex,2)) );
stats = regionprops(cc,'Area','PixelList'); 
for tempi = 1:length(stats)
    
   try %if stats(tempi).Area > 20
        
    %%%%% Find those nodes %%%%%
    indPxtempi = sub2ind( Df.imgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
    Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
    Lib = ismember(indPxNotNanAll,indPxtempi); [LibList,~] = find(Lib==1);
    
%     %%%%% Plane fitting %%%%%
%     uv = [U(2*notnanindex(LibList)-1), U(2*notnanindex(LibList))];
%     Rad = 1+2*winstepsize;
%     [~,UPlaneFittemp,FPlaneFittemp] = funCompDefGrad2(uv, coordinatesFEM(notnanindex(LibList),1:2), Rad, 1e6, dilatedI );
%     
%     % figure, plot3(coordinatesFEM(notnanindex(LibList),1), coordinatesFEM(notnanindex(LibList),2), uv(:,1), '.' );
%     % figure; plot3(coordinatesFEM(notnanindex(LibList),1), coordinatesFEM(notnanindex(LibList),2), uv(:,2), 'r.' );
%     % U(2*notnanindex(LibList)-1) = UPlaneFittemp(1:2:end);
%     % U(2*notnanindex(LibList)) = UPlaneFittemp(2:2:end);
%     
%     F(4*notnanindex(LibList)-3) = FPlaneFittemp(1:4:end);
%     F(4*notnanindex(LibList)-2) = FPlaneFittemp(2:4:end);
%     F(4*notnanindex(LibList)-1) = FPlaneFittemp(3:4:end);
%     F(4*notnanindex(LibList)-0) = FPlaneFittemp(4:4:end);
      
    % %%%%% RBF (Radial basis function) works better than "scatteredInterpolant" %%%%%
    % ------ Disp u ------
    op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[U(2*notnanindex(LibList)-1)]','RBFFunction', 'thinplate' ); %rbfcheck(op1);
    fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
    U(2*LiaList-1) = fi1(:);
    % figure, plot3(coordinatesFEM(notnanindex(LibList),1),coordinatesFEM(notnanindex(LibList),2),U(2*notnanindex(LibList)-1),'.')
    % hold on; plot3(coordinatesFEM(LiaList,1),coordinatesFEM(LiaList,2),fi1,'.')
    
    % ------ Disp v ------
    op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[U(2*notnanindex(LibList) )]','RBFFunction', 'thinplate' ); %rbfcheck(op1);
    fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
    U(2*LiaList ) = fi1(:);
    
    
    % if  (LocalICGNBadPtNum)/(size(coordinatesFEM,1)-length(row3)) < 0.4
    
        % ------ F11 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-3)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList-3) = fi1(:);
        % ------ F21 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-2)]','RBFFunction', 'thinplate');% rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList-2) = fi1(:);
        % ------ F12 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-1)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList-1) = fi1(:);
        % ------ F22 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-0)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList) = fi1(:);
    
    % end
    
    catch
    end
end


%%%%% Remove outliers %%%%%
% [~,F11RemoveOutlier] = rmoutliers(F(1:4:end), 'movmedian', 1+winstepsize);
% [~,F21RemoveOutlier] = rmoutliers(F(2:4:end), 'movmedian', 1+winstepsize);
% [~,F12RemoveOutlier] = rmoutliers(F(3:4:end), 'movmedian', 1+winstepsize);
% [~,F22RemoveOutlier] = rmoutliers(F(4:4:end), 'movmedian', 1+winstepsize);
% [F11RemoveOutlierInd,~] = find(F11RemoveOutlier==1);
% [F21RemoveOutlierInd,~] = find(F21RemoveOutlier==1);
% [F12RemoveOutlierInd,~] = find(F12RemoveOutlier==1);
% [F22RemoveOutlierInd,~] = find(F22RemoveOutlier==1);
% 
% for tempj=1:4
%     F(4*F11RemoveOutlierInd-4+tempj) = nan;
%     F(4*F21RemoveOutlierInd-4+tempj) = nan;
%     F(4*F12RemoveOutlierInd-4+tempj) = nan;
%     F(4*F22RemoveOutlierInd-4+tempj) = nan;
% end


% if (LocalICGNBadPtNum)/(size(coordinatesFEM,1)-length(row3)) > 0.39
%     
%     DICmeshtemp.coordinatesFEM = coordinatesFEM;
%     [F] = funGlobalNodalStrainRBF(DICmeshtemp,DICpara,U);
%     
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% RBF (Radial basis function) works better than "scatteredInterpolant" %%%%%
% op1 = rbfcreate( [coordinatesFEM(notnanindex,1:2)]',[U(2*notnanindex-1)]', 'RBFFunction', 'thinplate' ); rbfcheck(op1);
% fi1 = rbfinterp([coordinatesFEM(:,1:2)]', op1);
% % figure, plot3(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex-1),'.')
% % hold on; plot3(coordinatesFEM(:,1),coordinatesFEM(:,2),fi1,'.')
% 
% op2 = rbfcreate( [coordinatesFEM(notnanindex,1:2)]',[U(2*notnanindex)]','RBFFunction', 'thinplate'); rbfcheck(op2);
% fi2 = rbfinterp([coordinatesFEM(:,1:2)]', op2);
% U_rbf_thinplate = [fi1(:),fi2(:)]'; U = U_rbf_thinplate(:);
% 
% op = rbfcreate([coordinatesFEM(notnanindex,1:2)]', F(4*notnanindex-3)','RBFFunction', 'thinplate'); rbfcheck(op);
% fi11 = rbfinterp([coordinatesFEM(:,1:2)]', op );
% op = rbfcreate([coordinatesFEM(notnanindex,1:2)]', F(4*notnanindex-2)','RBFFunction', 'thinplate'); rbfcheck(op);
% fi21 = rbfinterp([coordinatesFEM(:,1:2)]', op );
% op = rbfcreate([coordinatesFEM(notnanindex,1:2)]', F(4*notnanindex-1)','RBFFunction', 'thinplate'); rbfcheck(op);
% fi12 = rbfinterp([coordinatesFEM(:,1:2)]', op );
% op = rbfcreate([coordinatesFEM(notnanindex,1:2)]', F(4*notnanindex-0)','RBFFunction', 'thinplate'); rbfcheck(op);
% fi22 = rbfinterp([coordinatesFEM(:,1:2)]', op );
% 
% F_rbf_thinplate = [fi11(:),fi21(:),fi12(:),fi22(:)]'; F = F_rbf_thinplate(:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Fill nans %%%%%%
nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);
nanindexF = find(isnan(F(1:4:end))==1); notnanindexF = setdiff([1:1:size(coordinatesFEM,1)],nanindexF);

% figure, plot3(coordinatesFEM(:,1),coordinatesFEM(:,2),F(4:4:end),'.');

if ~isempty(nanindex) || ~isempty(nanindexF)
    
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex-1),'natural','nearest');
U1 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex),'natural','nearest');
U2 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-3),'natural','nearest');
F11 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-2),'natural','nearest');
F21 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-1),'natural','nearest');
F12 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-0),'natural','nearest');
F22 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));

U = [U1(:),U2(:)]'; U = U(:);
F = [F11(:),F21(:),F12(:),F22(:)]'; F = F(:);



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== Use gridfit to interpolate ======
% Coordxnodes = unique(coordinatesFEM(:,1)); Coordynodes = unique(coordinatesFEM(:,2));
% [u1temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), U(2*notnanindex-1),Coordxnodes,Coordynodes,'regularizer','springs'); u1temp = u1temp';
% [v1temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), U(2*notnanindex),Coordxnodes,Coordynodes,'regularizer','springs'); v1temp = v1temp';
% [F11temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), F(4*notnanindex-3),Coordxnodes,Coordynodes,'regularizer','springs'); F11temp = F11temp';
% [F21temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), F(4*notnanindex-2),Coordxnodes,Coordynodes,'regularizer','springs'); F21temp = F21temp';
% [F12temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), F(4*notnanindex-1),Coordxnodes,Coordynodes,'regularizer','springs'); F12temp = F12temp';
% [F22temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), F(4*notnanindex-0),Coordxnodes,Coordynodes,'regularizer','springs'); F22temp = F22temp';
%   
% % Add remove outliers median-test based on: 
% % u1=cell(2,1); u1{1}=u1temp; u1{2}=v1temp;
% % [u2] = removeOutliersMedian(u1,4); u2temp=u2{1}; v2temp=u2{2};
% for tempi = 1:size(coordinatesFEM,1)
%     [row1,col1] = find(Coordxnodes==coordinatesFEM(tempi,1));
%     [row2,col2] = find(Coordynodes==coordinatesFEM(tempi,2));
%     U(2*tempi-1) = u1temp(row1,row2);
%     U(2*tempi)   = v1temp(row1,row2);
%     F(4*tempi-3) = F11temp(row1,row2); F(4*tempi-2) = F21temp(row1,row2);
%     F(4*tempi-1) = F12temp(row1,row2); F(4*tempi) = F22temp(row1,row2);
% end






