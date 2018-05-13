clear;clc;

%load the data
data_in = importdata('u.data');
user_id = data_in(:, 1);
item_id = data_in(:, 2);
rating = data_in(:, 3);

row_num = max(user_id);
col_num = max(item_id);

R = zeros(row_num, col_num);
W = zeros(row_num, col_num);

for i = 1:size(user_id)
    W(user_id(i), item_id(i)) = 1;
    R(user_id(i), item_id(i)) = rating(i);
end

L = 5;
N = size(item_id);
N = N(1);
%return a row vector containing a random permutation of the integers from 1 to N inclusive.
index = randperm(N)';
precision = zeros(1, 10);

for i = 0:9
    R_i = R;
    Q = zeros(row_num, col_num);
    I_record = zeros(row_num, col_num);
    index_i = index(i*(N/10)+1 : (i+1)*(N/10));
    for j = 1:size(index_i)
        R_i(user_id(index_i(j)), item_id(index_i(j))) = 0;
        Q(user_id(index_i(j)), item_id(index_i(j))) = 1;
    end
    [U_i, V_i] = wnmfrule1(R_i, 10);
    R_i_predict = U_i*V_i;
    R_i_predict = R_i_predict.*R.*Q;
    
    actual = 0;
    predict = 0;
    %sort elements of each row
    [B, I] = sort(R_i_predict, 2, 'descend');
    
    %record the top L movies recommended to each user
    for j = 1:row_num
        for k = 1:L
            I_record(j, I(j, k)) = 1;
        end
    end
    
    Predict_R = (I_record.*R_i_predict);
    for j = 1:row_num
        for k = 1:col_num
           if Predict_R(j, k) > 0
               predict = predict + 1;
               if R(j, k) >= 4
                   actual = actual + 1;
               end
           end
        end
    end
    
    precision(1, i+1) = actual/predict;
end
fprintf('The average precision is %f', sum(precision)/10);