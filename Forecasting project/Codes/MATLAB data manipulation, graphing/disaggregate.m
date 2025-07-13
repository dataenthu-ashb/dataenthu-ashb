function [vector] = disaggregate(dataPoint,cat)
  
    vector = zeros(3,1);

    if (cat == 1)
        %Summed quantities
        %Divide by 3
        entry = dataPoint / 3;
        vector(:,1) = entry;
    end
    if (cat == 2)
        %Percentage Change
        val = (dataPoint / 100) + 1;
        vector(:,1) = ((val^(1/3)) - 1) * 100;
    end
    if (cat == 3)
        %Slowly Increasing or decreasing
        vector(:,1) = dataPoint;
    end
    if (cat == 4)
        %Constant for extended periods and needs to adhere to specific
        %increment rule like basis points in terms of 0.25
        vector(:,1) = dataPoint;
    end
    if (cat == 6)
        %Indices or index-like values
        vector(:,1) = dataPoint;
    end
    if (cat == 7) 
        %Changes in large absolute values
        entry = dataPoint / 3;
        vector(:,1) = entry;
    end
    if (cat == 8)
        %Unique per period value
        entry = dataPoint / 3;
        vector(:,1) = entry;
    end

end