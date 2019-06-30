initiating =1;

% Function rolls_dice
% given a certain number of rolls, for a die of multiple sides,
% ... return the expected value of the random variable representing
% ... the value of a die roll.
% Also return, the average of all the rolls (as a comparison)
% As the number of rolls increases, the average_roll should
%... equal the expected_value.
function [average_roll,expected_value] = rolls_dice(rolls,sides)
    die_shape = [1:sides];
    trials = rolls;
    unbiased_probability = 1/length(die_shape);
    expected_value = sum(die_shape*unbiased_probability);
    outcomes = [];
    
    while trials > 0
        trials = trials-1;
        outcomes(rolls-trials) = die_shape(randi(length(die_shape)));
    end

    % prompt about your test
    desc = 'You rolled a %d-sided die, %d times';
    sprintf(desc,sides,rolls)
    average_roll = sum(outcomes)/length(outcomes);

    % show the rolls
    if length(outcomes) > 9
        'First 10 outcomes'
        outcomes(1:10)
    else
        'All outcomes'
        outcomes
    end


end

% rolls_dice(1,6)

[average,expected] = rolls_dice(2,6)
[average,expected] = rolls_dice(20,6)


[average,expected] = rolls_dice(10,12)
[average,expected] = rolls_dice(1000,12)