w_q = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0];
w_k = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0];
x = [0.9 0.1 0.1; 0.1 0.9 0.1; 0.5 0.5 0.1];
query = w_q * x;
key = w_k * x;
attention_score = query * key';

printf("query weights: \n")
disp(w_q)
printf("key weights: \n")
disp(w_k)
printf("input matrix: \n")
disp(x)
printf("query: \n");
disp(query);
printf("key: \n");
disp(key);
printf("attention score: \n");
disp(attention_score);

