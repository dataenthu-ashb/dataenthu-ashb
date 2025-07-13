%% Disaggregation Code
%99quarters
%Extract categories and data
array = table2array(Quarterly);
array(:,1) = [];
categories = array(1,:);
%%
categories = str2double(categories);
%%
data = array(2:end,:);

%%
data = str2double(data);
%%
n = length(data(:,1));
width = length(data(1,:));
disaggregatedData = zeros(1,width);

for i = 1:n
    dataRow = data(i,:);
    holderMatrix = zeros(3,width);
    for j = 1:width
        cat = categories(1,j);
        datapoint = dataRow(1,j);
        quarterofmonths = disaggregate(datapoint,cat);
        holderMatrix(:,j) = quarterofmonths;
    end
    disaggregatedData = [disaggregatedData;holderMatrix];
end

disaggregatedData(1,:) = [];

%%
Lidx=find(imag(disaggregatedData)~=0);
%%
filename = 'QtoM-EUROPE-LSTM.xlsx';
writematrix(disaggregatedData,filename,'Sheet',1);

