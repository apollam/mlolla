## Mudanças para v2

- [x] Organizar estrutura (PRINCIPAL)
- [x] Adicionar readme em todas as pastas
- [x] Remover arquivos não utilizados e colocar em algum backup: S3, fairness metrics, 
multiclass, transformers, etc  
- [x] Adicionar fairness evaluation
- [x] Adicionar multiclass evaluation
- [x] Adicionar sessão de data_exploration contendo plots e prints de informações
- [x] Fazer projeto identificar automaticamente se é regressão, classificação ou multi
- [ ] Melhorar data_acquisition folder
    - [x] read xlsb local
    - [x] read xlsx local 
    - [x] read csv local 
    - [x] connect and read from sql
    - [ ] unificar todos em uma função só

   
## Backlog
- [ ] Ajustar os imports
- [ ] Melhorar transformers do Preprocessing
    - [x] Adicionar mais
    - [ ] Verificar se funcionam
    - [ ] Adicionar tests (adicionar pytest)
- [ ] Adicionar data validation 
(TDDA ou https://analyticsindiamag.com/6-python-data-validating-tools-to-use-in-2019/)
- [ ] Adicionar output de graficos
- [ ] Adicionar predictions validation
- [ ] Upar resultados em algum DB ou framework
- [ ] Adicionar licença 
(https://stackoverflow.com/questions/31639059/how-to-add-license-to-an-existing-github-project)
- [ ] Adicionar experimentation control
- [ ] Adicionar dockerização
- [ ] Fazer diagrama com a organização das pastas e que classe puxa que classe
- [ ] Adicionar model interpretation (ver como o fklearn faz)
- [ ] Mudar autoria dos commits
- [ ] Adicionar documentação de cada função e classe
- [ ] Time series 
- [ ] Scores for label 0
- [ ] Outlier detection transformer
- [ ] Adicionar integração com AWS/Azure
- [ ] Melhorar chamada do trainer dependendo do learning_type (tem muita linha)
- [ ] Organizar melhor o trainer
- [ ] Adicionar novos parametros: n_iter, kfold diferentes, metricas diferentes 
- [ ] Agrupar todos os READMEs em um só
- [ ] Adicionar tratamento de erros e um sistema de log
- [ ] Criar classes abstratas de Trainer e Metrics
- [ ] Criar *possibilidade* de setar os algoritmos de maneira alto nível

