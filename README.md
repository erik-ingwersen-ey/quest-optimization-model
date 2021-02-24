[![deepcode](https://www.deepcode.ai/api/gh/badge?key=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwbGF0Zm9ybTEiOiJnaCIsIm93bmVyMSI6ImluZ3dlcnNlbjk2IiwicmVwbzEiOiJFWS1RdWVzdC1EaWFnbm9zdGljcyIsImluY2x1ZGVMaW50IjpmYWxzZSwiYXV0aG9ySWQiOjEzMzIwLCJpYXQiOjE2MDUxOTU5NDh9.GY0cs_K39qlFuf5meK25js1OJAGhBCqTwWAFwSJPduc)](https://www.deepcode.ai/app/gh/ingwersen96/EY-Quest-Diagnostics/_/dashboard?utm_content=gh%2Fingwersen96%2FEY-Quest-Diagnostics) ![version](https://badgen.net/badge/Version/V1.8/green?icon=github). [![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=ingwersen96_EY-Quest-Diagnostics&metric=sqale_rating&token=afea3e7dc8e10faf0d1ee50e391e00d04254f556)](https://sonarcloud.io/dashboard?id=ingwersen96_EY-Quest-Diagnostics)
[![Sonarcloud Code Quality Test](https://github.com/ingwersen-erik/quest-optimization-model/actions/workflows/build.yml/badge.svg?branch=main&event=status)](https://github.com/ingwersen-erik/quest-optimization-model/actions/workflows/build.yml)


<p align="left">
  <img src="https://github.com/ingwersen96/ANBIMA-Cyber/blob/master/Github/Gifs/EY-animated-logo.gif">
</p>


# EY - Quest Diagnostics

Repositório para armazenamento dos códigos fonte do projeto de otimização de inventário para a Quest Diagnóstics.

# Instruções

## Clonar o Repositório

Para criar uma cópia do repositório na sua maquina local, você precisa abrir uma aba do **Command Prompt**. Depois de aberta, ir para o diretório no qual se deseja colocar o repositório:

```bash
C:\Users\YG287PG>cd Desktop
```

Depois, para clonar o repositório, basta copiar o seguinte comando no **Command Prompt**:

```bash
C:\Users\YG287PG\Desktop>git clone https://github.com/ingwersen96/EY-Quest-Diagnostics.git
```
**Output:**
```bash
Cloning into 'EY-Quest-Diagnostics'...
remote: Enumerating objects: 17, done.
remote: Counting objects: 100% (17/17), done.
remote: Compressing objects: 100% (13/13), done.
remote: Total 17 (delta 2), reused 0 (delta 0), pack-reused 0
Receiving objects: 100% (17/17), 4.51 KiB | 115.00 KiB/s, done.
Resolving deltas: 100% (2/2), done.
```

## Adicionar Novos Códigos para o Repositório Remoto

**Passo 1:** Adicionar novos arquivos ao stagging
Para adicionar novos arquivos no repositório remoto, primeiramente é necessário adicionar no stagging alterações no repositório. Você pode fazer isso com o comando `git add [file]`. 

>**Observação:** Se os arquivos que você quiser adicionar ao repositório remoto já existirem, ou sejam se você quiser somente atualizar os arquivos que estão lá, você pode pular para o próximo passo.

**Exemplo:**
```bash
git add DataPrep.py  Solution_Space.py main.py  requirements.txt  runtime.py
```
**Passo 2:** Realizar o commit.
Você pode realizar commits utilizando o comando `git commit -m “[mensagem descritiva]”`.
>**Observação:** A adição de uma mensagem descrevendo o commit é obrigatória. O comando não funcionará sem ela.

**Exemplo:**
```bash
git commit -m "Optimization files"
```
**Output:**
```bash
[main 9ffda2d] Optimization files
 5 files changed, 873 insertions(+)
 create mode 100644 DataPrep.py
 create mode 100644 Solution_Space.py
 create mode 100644 main.py
 create mode 100644 requirements.txt
 create mode 100644 runtime.py
```

**Passo 3:** Enviar para o repositório remoto
Por fim, você precisa fazer o upload dos arquivos "commitados".

```bash
git push
```
**Output:**
```bash
Enumerating objects: 8, done.
Counting objects: 100% (8/8), done.
Delta compression using up to 4 threads
Compressing objects: 100% (6/6), done.
Writing objects: 100% (7/7), 9.71 KiB | 310.00 KiB/s, done.
Total 7 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/ingwersen96/EY-Quest-Diagnostics.git
   066beae..9ffda2d  main -> main
```

## Github

Marcelo, como o projeto é relativamente pequeno em termos de equipe, criei de maneira sugestiva, **2 branches** que podemos usar ao longo do desenvolvimento:

* **main:** Para versões estáveis de código. 
* **dev:** sugiro irmos usando essa branch ao longo do desenvolvimento.

A ideia é após algumas iterações na branch **dev**, nós fazermos um merge para passarmos o código produzido para a branch **main**. Com isso, sempre teremos uma versão estável, que podemos apresentar ao cliente e outra para irmos fazendo os testes no modelo.

## Sonar

Sonarqube é uma ferramenta que realiza análises de qualidade nos códigos-fonte de projetos. Configurei a ferramenta para realizar análises de qualidade de código toda vez que novos códigos são adicionados na branch **main**. O output da ferramenta é um relatório com possíveis melhorias no código fonte identificadas. Não precisamos seguir as melhorias, apenas configurei a ferramenta de maneira sugestiva.
