function [agData] = aggregate(dataSegment,categories)
    
    width = length(dataSegment(1,:));
    height = length(dataSegment(:,1));
    agData = zeros(1,width);

    for w = 1:width
        Category = categories(1,w);
        dataColumn = dataSegment(:,w);
        if (Category == 1)
            %Summed quantities
            agData(1,w) = sum(dataColumn);
        end
        if (Category == 2) %Check
            %Percentage Change
            vec = (dataColumn/100) + 1;
            agData(1,w) = (prod(vec) - 1) * 100;
        end
        if (Category == 3) %Check
            %Slowly Increasing or decreasing
            agData(1,w) = mean(dataColumn);
        end
        if (Category == 4) %Check
            %Constant for extended periods and needs to adhere to specific
            %increment rule like basis points in terms of 0.25
            agData(1,w) = dataColumn(end);
        end
        if (Category == 6) %Done
            %Indices or index-like values
            agData(1,w) = mean(dataColumn);
        end
        if (Category == 7) %Check
            %Changes in large absolute values
            agData(1,w) = sum(dataColumn);
        end
        if (Category == 8) %Check
            %Unique per period value
            agData(1,w) = sum(dataColumn);
        end 
    end
end