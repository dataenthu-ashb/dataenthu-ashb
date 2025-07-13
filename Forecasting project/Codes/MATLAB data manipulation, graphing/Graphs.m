%% Graphs
%% DATES
datesQuarter = table2array(datesQuarter);
datesMonth = table2array(datesMonth);
%%
datesQuarter = datetime(datesQuarter,"Format","uuuu-QQQ");
%%
datesMonth = datetime(datesMonth,'Format','dd-mmm-uuuu');
%% EU GDP 
EU_GDP = table2array(EU_GDP);
EU_PCE = table2array(EU_PCE);
EU_UNEMP = table2array(EU_UNEMP);
US_GDP = table2array(US_GDP);
US_PCE = table2array(US_PCE);
US_UNEMP = table2array(US_UNEMP);
%%
figure
%Set 1 - GDP 
tiledlayout(1,2)
%Tile 1 EU GDP
nexttile
p = plot(EU_GDP);
%ACTUAL STYLE
p(1).LineWidth = 2;
p(1).Marker = '.';
p(1).MarkerFaceColor = [0 0.4470 0.7410];
p(1).Color = [0 0.4470 0.7410];
%ARIMAX STYLE
p(2).Marker = '.';
p(2).MarkerFaceColor = [0.8500 0.3250 0.0980];
p(2).Color = [0.8500 0.3250 0.0980];
%MIDAS STYLE
p(3).Marker = '.';
p(3).MarkerFaceColor = [0.9290 0.6940 0.1250];
p(3).Color = [0.9290 0.6940 0.1250];
%LSTM STYLE
p(4).Marker = '.';
p(4).MarkerFaceColor = [0.4660 0.6740 0.1880];
p(4).Color = [0.4660 0.6740 0.1880];
%ARIMAX LSTM HYBRID STYLE
p(5).Marker = '.';
p(5).MarkerFaceColor = [0.3010 0.7450 0.9330];
p(5).Color = [0.3010 0.7450 0.9330];
%MIDAS LSTM HYBRID STYLE
p(6).Marker = '.';
p(6).MarkerFaceColor = [1 0 0];
p(6).Color = [1 0 0];

xlabel('Observation Number');
ylabel('% Change in GDP for EU');
grid off;

%Tile 2 US GDP
nexttile
p = plot(US_GDP);
%ACTUAL STYLE
p(1).LineWidth = 2;
p(1).Marker = '.';
p(1).MarkerFaceColor = [0 0.4470 0.7410];
p(1).Color = [0 0.4470 0.7410];
%ARIMAX STYLE
p(2).Marker = '.';
p(2).MarkerFaceColor = [0.8500 0.3250 0.0980];
p(2).Color = [0.8500 0.3250 0.0980];
%MIDAS STYLE
p(3).Marker = '.';
p(3).MarkerFaceColor = [0.9290 0.6940 0.1250];
p(3).Color = [0.9290 0.6940 0.1250];
%LSTM STYLE
p(4).Marker = '.';
p(4).MarkerFaceColor = [0.4660 0.6740 0.1880];
p(4).Color = [0.4660 0.6740 0.1880];
%ARIMAX LSTM HYBRID STYLE
p(5).Marker = '.';
p(5).MarkerFaceColor = [0.3010 0.7450 0.9330];
p(5).Color = [0.3010 0.7450 0.9330];
%MIDAS LSTM HYBRID STYLE
p(6).Marker = '.';
p(6).MarkerFaceColor = [1 0 0];
p(6).Color = [1 0 0];

legend("ACTUAL","ARIMAX","MIDAS","LSTM","HYBRID ARIMAX-LSTM","HYBRID MIDAS-LSTM","Location","best")
xlabel('Observation Number');
ylabel('% Change in GDP for US');
grid off;

%%
figure
%Set 2 - PCE 
tiledlayout(1,2)

%Tile 1 EU PCE
nexttile
p = plot(EU_PCE);
%ACTUAL STYLE
p(1).LineWidth = 2;
p(1).Marker = '.';
p(1).MarkerFaceColor = [0 0.4470 0.7410];
p(1).Color = [0 0.4470 0.7410];
%ARIMAX STYLE
p(2).Marker = '.';
p(2).MarkerFaceColor = [0.8500 0.3250 0.0980];
p(2).Color = [0.8500 0.3250 0.0980];
%MIDAS STYLE
p(3).Marker = '.';
p(3).MarkerFaceColor = [0.9290 0.6940 0.1250];
p(3).Color = [0.9290 0.6940 0.1250];
%LSTM STYLE
p(4).Marker = '.';
p(4).MarkerFaceColor = [0.4660 0.6740 0.1880];
p(4).Color = [0.4660 0.6740 0.1880];
%ARIMAX LSTM HYBRID STYLE
p(5).Marker = '.';
p(5).MarkerFaceColor = [0.3010 0.7450 0.9330];
p(5).Color = [0.3010 0.7450 0.9330];
%MIDAS LSTM HYBRID STYLE
p(6).Marker = '.';
p(6).MarkerFaceColor = [1 0 0];
p(6).Color = [1 0 0];

xlabel('Observation Number');
ylabel('% Change in PCE for EU');
grid off;

%Tile 2 US PCE
nexttile
p = plot(US_PCE);
%ACTUAL STYLE
p(1).LineWidth = 2;
p(1).Marker = '.';
p(1).MarkerFaceColor = [0 0.4470 0.7410];
p(1).Color = [0 0.4470 0.7410];
%ARIMAX STYLE
p(2).Marker = '.';
p(2).MarkerFaceColor = [0.8500 0.3250 0.0980];
p(2).Color = [0.8500 0.3250 0.0980];
%MIDAS STYLE
p(3).Marker = '.';
p(3).MarkerFaceColor = [0.9290 0.6940 0.1250];
p(3).Color = [0.9290 0.6940 0.1250];
%LSTM STYLE
p(4).Marker = '.';
p(4).MarkerFaceColor = [0.4660 0.6740 0.1880];
p(4).Color = [0.4660 0.6740 0.1880];
%ARIMAX LSTM HYBRID STYLE
p(5).Marker = '.';
p(5).MarkerFaceColor = [0.3010 0.7450 0.9330];
p(5).Color = [0.3010 0.7450 0.9330];
%MIDAS LSTM HYBRID STYLE
p(6).Marker = '.';
p(6).MarkerFaceColor = [1 0 0];
p(6).Color = [1 0 0];

legend("ACTUAL","ARIMAX","MIDAS","LSTM","HYBRID ARIMAX-LSTM","HYBRID MIDAS-LSTM", "Location","best")
xlabel('Observation Number');
ylabel('% Change in PCE for US');
grid off;
%%
figure
%Set 3 - UNEMP 
tiledlayout(1,2)

%Tile 1 EU UNEMP
nexttile
p = plot(EU_UNEMP);
%ACTUAL STYLE
p(1).LineWidth = 2;
p(1).Marker = '.';
p(1).MarkerFaceColor = [0 0.4470 0.7410];
p(1).Color = [0 0.4470 0.7410];
%ARIMAX STYLE
p(2).Marker = '.';
p(2).MarkerFaceColor = [0.8500 0.3250 0.0980];
p(2).Color = [0.8500 0.3250 0.0980];
%MIDAS STYLE
p(3).Marker = '.';
p(3).MarkerFaceColor = [0.9290 0.6940 0.1250];
p(3).Color = [0.9290 0.6940 0.1250];
%LSTM STYLE
p(4).Marker = '.';
p(4).MarkerFaceColor = [0.4660 0.6740 0.1880];
p(4).Color = [0.4660 0.6740 0.1880];
%ARIMAX LSTM HYBRID STYLE
p(5).Marker = '.';
p(5).MarkerFaceColor = [0.3010 0.7450 0.9330];
p(5).Color = [0.3010 0.7450 0.9330];
%MIDAS LSTM HYBRID STYLE
p(6).Marker = '.';
p(6).MarkerFaceColor = [1 0 0];
p(6).Color = [1 0 0];

xlabel('Observation Number');
ylabel('% Change in UNEMP for EU');
grid off;

%Tile 2 US UNEMP
nexttile
p = plot(US_UNEMP);
%ACTUAL STYLE
p(1).LineWidth = 2;
p(1).Marker = '.';
p(1).MarkerFaceColor = [0 0.4470 0.7410];
p(1).Color = [0 0.4470 0.7410];
%ARIMAX STYLE
p(2).Marker = '.';
p(2).MarkerFaceColor = [0.8500 0.3250 0.0980];
p(2).Color = [0.8500 0.3250 0.0980];
%MIDAS STYLE
p(3).Marker = '.';
p(3).MarkerFaceColor = [0.9290 0.6940 0.1250];
p(3).Color = [0.9290 0.6940 0.1250];
%LSTM STYLE
p(4).Marker = '.';
p(4).MarkerFaceColor = [0.4660 0.6740 0.1880];
p(4).Color = [0.4660 0.6740 0.1880];
%ARIMAX LSTM HYBRID STYLE
p(5).Marker = '.';
p(5).MarkerFaceColor = [0.3010 0.7450 0.9330];
p(5).Color = [0.3010 0.7450 0.9330];
%MIDAS LSTM HYBRID STYLE
p(6).Marker = '.';
p(6).MarkerFaceColor = [1 0 0];
p(6).Color = [1 0 0];

legend("ACTUAL","ARIMAX","MIDAS","LSTM","HYBRID ARIMAX-LSTM","HYBRID MIDAS-LSTM")
xlabel('Observation Number');
ylabel('% Change in UNEMP for US');
grid off;
%%