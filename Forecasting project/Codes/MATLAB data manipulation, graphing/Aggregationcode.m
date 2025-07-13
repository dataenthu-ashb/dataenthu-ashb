% Aggregation Disaggregation Main
%% 
Dates = Daily(2:end,1);

%%
dates = convertCharsToStrings(Dates);
dates = table2array(dates);
%%
dates = datetime(dates,"InputFormat",'uuuu-MM-dd');

%%
datesAsMonths = datetime(dates,"Format","MMM-uuuu");
%%
data2 = Daily(1:end,2:end);
%%
data = table2array(data2);
%%
%data(:,1) = [];
aggregated = data(1,:);
%%
data = str2double(data);
aggregated = str2double(aggregated);
%%
datesAsMonths = string(datesAsMonths);
%%
n = length(datesAsMonths);
lastdate = datesAsMonths(1);
filtereddates = string(zeros(1));
filtereddates(1) = lastdate;
%%
data(1,:) = [];
%%
for i = 1:n
    currentdate = datesAsMonths(i);
    if(not(currentdate == lastdate))
        filtereddates(end + 1,1) = currentdate;
    end
    lastdate = currentdate;
end
%%
m = length(filtereddates);
n = length(datesAsMonths);
width = length(data(1,:));
aggregatedData = zeros(1,width);

for j = 1:m
    targetDate = filtereddates(j);
    dataSegment = zeros(1,width);
    
    for i = 1:n
        if (datesAsMonths(i) == targetDate)
            dataSegment(end + 1, :) = data(i,:);
        end
    end
    
    dataSegment(1,:) = [];

    agData = aggregate(dataSegment,aggregated);
    aggregatedData(end + 1, :) = agData(1,:);

end
aggregatedData(1,:) = [];

%%
filename = 'DtoM-EUROPE-LSTM.xlsx';
writematrix(aggregatedData,filename,'Sheet',1);
