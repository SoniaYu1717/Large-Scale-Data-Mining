% load data
A = importdata('u.data');
user_id = A(:, 1);
item_id = A(:, 2);
rating = A(:, 3);

% problem1:matrix factorization
% there are 943 users and 1682 items in the dataset
R = zeros(943, 1682);
W = zeros(943, 1682);
for i=1:100000
    R(user_id(i), item_id(i)) = rating(i);
    W(user_id(i), item_id(i)) = 1;
end

index = randperm(100000);
avg_error = zeros(10, 3);

for i=1:10
%     R_tmp = zeros(943, 1682);
%     W_tmp = zeros(943, 1682);
    R_tmp = R;
    W_tmp = W;
%     for j=1:(i-1)*10000
%         R_tmp(user_id(index(j)), item_id(index(j))) = rating(index(j));
%         W_tmp(user_id(index(j)), item_id(index(j))) = 1;
%     end
    
    for j=((i-1)*10000+1):(i*10000)
        R_tmp(user_id(index(j)), item_id(index(j))) = 0;
    end
    
%     for j=(i*10000+1):100000
%         R_tmp(user_id(index(j)), item_id(index(j))) = rating(index(j));
%         W_tmp(user_id(index(j)), item_id(index(j))) = 1;
%     end
    
    for j=1:3
        k=[10, 50, 100];
        [U_tmp, V_tmp] = wnmfrule(R_tmp, k(j), W_tmp);
        R_predict = U_tmp * V_tmp;
        error_tmp = 0;
        for m = ((i-1)*10000+1) : (i*10000)
            error_tmp = error_tmp + abs(R_predict(user_id(index(m)), item_id(index(m))) - R(user_id(index(m)), item_id(index(m))));
        end
        error_tmp = error_tmp/10000;
        avg_error(i, j) = error_tmp;
    end
end






