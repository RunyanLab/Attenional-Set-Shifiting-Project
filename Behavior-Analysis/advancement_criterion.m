%% load mouse behavior session 

mouse_Name = 'CBEI2'
date = 211129
load ([mouse_Name '_' num2str(date) '_Cell' '.mat'])
%%load CBEI2_211117_Cell.mat

%% plot 30 trial moving average for total correct

for a = 1:length(dataCell)
    x(a) = dataCell{a}.result.correct
    %turn(a) = dataCell{a}.result.leftTurn
end

M = movmean(x,30)
%T = movmean(turn,30)
lin1 = zeros(1,length(x))+0.8; %creates a line at 80% accuracy
plot(M)
hold on
plot(lin1,'k') %plots criterion line
hold off

%% calculate the percentage of a session the mouse spent over the advancement criterion requirement 

num = sum(M >= .80) %% goes through M and finds all trials over the 80% criterion line 
perc_M = num / length(dataCell) 
perc_M
clear
%% peak moving average 

for date = [211115:211117]
    load ([mouse_Name '_' num2str(date) '_Cell' '.mat'])
    for a = 1:length(dataCell)
        x(a) = dataCell{a}.result.correct
    end
    M(date) = max(movmean(x,30))
end

%% plot 30 trial moving average for left and right trials 

for a = 1:length(dataCell);
    corr(a) = dataCell{a}.result.correct;
    left_turn(a) = dataCell{a}.result.leftTurn;
    if left_turn(a) ~= 1 && corr(a) <= 1;
        trial_type(a) = 2 %%left 
    elseif left_turn(a) ~= 0 && corr(a) <= 1;
        trial_type(a) = 3 %%right
    end
end

corr_matrix = [corr' trial_type']
%corr_trials = corr' + trial_type'

right = corr_matrix(corr_matrix(:, 2)== 2, :) %sort by right trials 
left = corr_matrix(corr_matrix(:, 2)== 3, :) %sort by left trials 
lin2 = zeros(1,length(right))+0.5; %creates a line at 50% accuracy

plot(movmean(right(:,1),30))
hold on 
plot(movmean(left(:,1),30))
plot(lin2,'k') %plots criterion line
legend('right','left')
