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

%using wnmfrule, get the final U and V matrices, and residual
[U1, V1, ~, ~, r1] = wnmfrule(R, 10, W);
[U2, V2, ~, ~, r2] = wnmfrule(R, 50, W);
[U3, V3, ~, ~, r3] = wnmfrule(R, 100, W);

fprintf('k=10, total LSE: %f\n', r1);
fprintf('k=50, total LSE: %f\n', r2);
fprintf('k=100, total LSE: %f\n', r3);




