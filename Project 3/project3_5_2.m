clear; clc;

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
    R(user_id(i), item_id(i)) = rating(i);
    W(user_id(i), item_id(i)) = 1;
end

L = 50;
N = size(item_id);
N = N(1);
index = randperm(N)';
hit_L = zeros(1, L);
false_alarm_L = zeros(1, L);

M_L = zeros(10, L);
N_L = zeros(10, L);

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
    R_test = R.*Q;
        
%compute actual like
    actual_t = 0;
    actual_f = 0;
    for j = 1:row_num
        for k = 1:col_num
            if R_test(j, k) > 0
                if R_test(j, k) >= 4
                    actual_t = actual_t + 1; % the number of test movies liked by the users
                else
                    actual_f = actual_f + 1; % the number of test movies not liked by the users
                end
            end
        end
    end

    for t = 1:L
    %compute suggested true and false
        [B, I] = sort(R_i_predict, 2, 'descend');
        predict_t = 0;
        predict_f = 0;
        for j = 1:row_num
            for k = 1:t
                I_record(j, I(j, k)) = 1;
            end
        end
    
        Predict_R = (I_record.*R_i_predict);
        for j = 1:row_num
            for k = 1:col_num
                if Predict_R(j, k) > 0
                    if R_test(j, k) >= 4
                        predict_t = predict_t + 1; % the number of test movies liked by the users and suggested by the algorithm
                    else
                        if R_test(j,k) >0
                            predict_f = predict_f + 1; % the number of test movies not liked by the users but suggested by the algorithm
                        end
                    end
                end
            end
        end

        M_L(i+1, t) = predict_t /actual_t;
        N_L(i+1, t) = predict_f /actual_f;
    end
end

for t = 1:L
    tmp1 = 0;
    tmp2 = 0;
    for i = 1:10
        tmp1 = tmp1 + M_L(i, t);
        tmp2 = tmp2 + N_L(i, t);
    end
    hit_L(1,t) = tmp1/10;
    false_alarm_L(1,t) = tmp2/10;
end

figure(1);
plot(false_alarm_L, hit_L, 'r', 'LineWidth',2);
xlabel('False-alarm Rate');
ylabel('Hit Rate');
title('Hit Rate vs. False-alarm Rate');