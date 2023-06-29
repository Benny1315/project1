% A filter that works according to the average of the nearest members.
% Very similar to a moving average.
% Not used, but tested and failed with noisy signals.

function G = DeNoising(Y)
nirmul = 1/4;
for i = 1:length(Y(:,1))
for j = 1:2
    if (Y(i,1)>1.1)   % lines 6-20 to normalize the first member.
        Y(i,1) = (Y(i,2) + Y(i,3)...
            + Y(i,4) + Y(i,5))*nirmul;
         if (Y(i,1)>1.1)
            Y(i,:) = 0;
            break
        end
    elseif(Y(i,1)<-0.2)
        Y(i,1) = (Y(i,2) + Y(i,3)...
            + Y(i,4) + Y(i,5))*nirmul;
         if (Y(i,1)<-0.2)
            Y(i,:) = 0;
            break
        end
    end
    if (Y(i,2)>1.1)   % lines 21-35 to normalize the second member.
        Y(i,2) = (Y(i,1) + Y(i,3)...
            + Y(i,4) + Y(i,5))*nirmul;
         if (Y(i,2)>1.1)
            Y(i,:) = 0;
            break
        end
    elseif(Y(i,2)<-0.2)
        Y(i,2) = (Y(i,1) + Y(i,3)...
            + Y(i,4) + Y(i,5))*nirmul;
         if (Y(i,2)<-0.2)
            Y(i,:) = 0;
            break
        end
    end
for j = 3:length(Y(1,:))-4
    if (Y(i,j)>1.1)
        Y(i,j) = (Y(i,j-1) + Y(i,j-2)...
            + Y(i,j+1) + Y(i,j+2))*nirmul;
        if (Y(i,j)>1.1)
            Y(i,:) = 0;
            break
        end
    elseif (Y(i,j)<-0.2)
        Y(i,j) = (Y(i,j-1) + Y(i,j-2)...
            + Y(i,j+1) + Y(i,j+2))*nirmul;
        if (Y(i,j)<-0.2)
            Y(i,:) = 0;
            break
        end
    end
end
end
end
Y(~any(Y,2), : ) = [];
G = Y;
end
