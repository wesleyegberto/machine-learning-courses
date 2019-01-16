# Linguagem Octave-Matlab

* Comandos que terminam em ; não são printados
* Podemos agrupar vários comandos usando ,
  * a = 1, b = 2, c = 3;
* Comentários iniciam com %
  


## Ajuda
help <funcao>
who % mostrar as variaveis
whos % mostrar as variáveis com detalhes como tamanho em bytes, tipo e dimensões
clear % remover todas as variáveis

disp('') % exibir mensagem

## Vetores e Matrizes
* 1-based index
vector = [1 2 3] % matriz 1x3
vector_transpose = [1; 2; 3] % matriz 3x1
matrix = [1 2 3; 4 5 6; 7 8 9]

* Gerar matrizes com 1's ou 0's
matrix_ones = ones(3, 3)
matriz_twos = 2 * ones(3, 3)
matriz_zeros = zeros(3, 3)

* Gerar matrizes com valores aleatórios
matriz_rands = rand(3, 3)
* Gerar matrizes com valores gaussianos
matriz_gaussian = randn(3, 3)
* Gerar matriz identidade
matriz_identity = eye(3, 3)

* Dimensões de matrizes
sz = size(ones(3, 2)) % retorna um vetor com as dimensões - [3, 2]
s1 = size(ones(3, 2), 1) % return 3
s2 = size(ones(3, 2), 2) % return 2
l = length(ones(3, 2)) % return 4 - maior dimensão

* Operações de leitura de matrizes
table = ones(5, 5)
value = table(3, 2) % linha 3 coluna 2
row_2 = table(2, :) % toda linha 2
col_3 = table(:, 3) % toda coluna 3
rows_1_3 = table([1, 3], :) % linhas 1 e 3
col_1_first_10 = tables(1:10) % copia os 10 primeiros valores da coluna 1
elements = table(:) % retorna um vetor com todos elementos

* Operações de escrita
table(1, 1) = 3
table(2, :) = [1 2 3 4 5] % atualiza a linha inteira
table(:, 1) = [1; 2; 3; 4; 5] % atualiza a coluna inteira

* Concatenar matrizes horizontalmente (uma ao lado da outra)
C = [A B] % mesma quantidade de linhas

* Concatenar matrizes verticalmente (uma em baixo da outra)
C = [A; B] % mesma quantidad de colunas

* Produto de matrizes
C = A * B

* Operações element-wise (entre os elementos de índices correspondentes)
C = A .* B % multiplicando
C = A .^ B % potenciação
C = A ./ B
C = -A % multiplicação por -1

B = A < 3 % retorna 1 para true e 0 para false
i = find(a < 3) % retorna os indices dos elems < 3
[i1, i2] = find(A < 3) % retorna os índices (j e k) dos elems < 3

* Funções com matriz
  * Se não passar segundo parâmetro então aplica element-wise
  * Se segundo parâmetro for 1 aplica column-wise
  * Se segundo parâmetro for 2 aplica row-wise
  
C = log(A) % logarítmo
log_cols = log(A, 1)
log_rows = log(A, 2)
C = exp(A) % potenciação com base e (e ^ elemento)
C = abs(A) % valor absoluto (sem sinal)
S = sum(A) % soma todos os elementos
sum_cols = sum(A, 1)
sum_rows = sum(A, 2)
P = prod(A) % multiplica todos os elemenos

C = floor(A) % arredonda para baixo
C = ceil(A) % arredonda para cima

* Transposição de matrizes
v = [1 2 3]
v_t = v' % transposição do vetor v (matriz 1x3)
A = [1 2; 3 4; 5 6]
B = A' % transposição da matriz

* Matriz inversa
A_i = pinv(A)

* Encontrar o maior valor
v = [4 5 0]
max = max(v) % retorna 5
[max, ind] = max(v) % retorna 5 e a posição 2
max_per_col = max(A, [], 1) = max(A) % retornar um vetor com os maiores por coluna
mex_per_row = max(A, [], 2) % retornar um vetor com os maiores por linha

max(max(A)) % maior de toda matriz
max(A(:)) % maior de toda a matriz usando o vetor de todos os elems



### Controle de Fluxo
* For
for i=1:10,
  disp(i);
end

* While
i = 0;
while i<10,
  disp(i);
  i = i+1;
end



#### Funções
* Sem retorno
function showSomething()
	disp('Void');
end

* Função que retorna um valor
function x = myFunction()
	x = 10
end

* Função que retorna dois valores
function [x, y] = myValues()
	x = 1
	y = 2
end



## Gráficos
* Plotar multiplos gráficos usamos
hold on % ativar
hold off % desativar

* Customização
xlabel('<label axis x>')
ylabel('<label axis y>')
legend('data A', 'data B') % legenda para cada gráfico
title('Titulo qualquer')
axis([0 5 -1 1]) % escala de x (0 a 5) e y (-1 a 1)

* Salvar gráfico
print -dpng 'my_graph.png'

* Abrir vários gráficos
figure(1)
figure(2)

* Figura com dois gráficos
subplot(1, 2, 1) % cria tela com dois gráficos (1x2) e acessa o primeiro (1)
plot(x, y) % plota no gráfico 1
subplot(1, 2, 2) % acessa o segundo (2)
plot(x, y) % plota no gráfico 2

* Limpar
clf

* Fechar gráfico
close

* Plotar matriz
imagesc(A) % escala de cores aleatória para cada valor
imagesc(A), colorbar, colormap gray; % escala de cores cinza (menor valor é preto)


#### Gráfico x e y
* Plotar gráfico com X e Y e escolhendo cor
x = [1 2 3 4 5]
y = sin(pi * x)
plot(x, y, 'r') % r -> red


#### Histograma
w = -6 + 3.14 * randn(1, 1000)
hist(w)
hist(w, 50)



## Arquivos

#### Carregar arquivo
% ao carregar é criado uma variável com o mesmo nome para armazenar os dados
load <nome_arquivo> 
load('<nome_arquivo>')

% salvar uma variável em arquivo - quando der load irá recarregá-la

% salvar formato binário
save <nome_arquivo> <variavel>

% salvar formato ASCII
save <nome_arquivo> <variavel> -ascii



## Funções Padrões

* std: calcular o desvio padrão
* mean: calcular média aritmética
