function AvgAhuzP = AvgPowerCal(before, after)
Pbefore = sum(before(:,:).^2,2);
Pafter = sum(after(:,:).^2,2);
AhuzP = (abs(Pafter)./Pbefore)*100;
AvgAhuzP = mean(AhuzP);
end