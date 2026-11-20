# TCC: inferência distribuída ao vivo com 20 nós

Perfil isolado do experimento RunPod de 50 nós. O Terraform lê o mesmo perfil do
executor e faz um **plano local, sem provider nem recursos remotos**: apenas confere
a topologia, o modelo e o volume de trabalho. Não cria máquina nem altera a
campanha preexistente. Não é um plano de compra/provisionamento.

## Grade e inferência

| Cenário | Nós em conluio | Transformação |
| --- | ---: | --- |
| baseline | 0/20 | nenhuma |
| m20-alto | 4/20 | ruído multiplicativo forte |
| m40-alto | 8/20 | ruído multiplicativo forte |
| m80-alto | 16/20 | ruído multiplicativo forte |
| m40-sutil | 8/20 | distância relativa 0,2 |
| m80-sutil | 16/20 | distância relativa 0,2 |

Cada cenário usa 50 questões fixas do GSM8K e cinco sementes (0–4), um cenário e
uma semente por disparo, 1.000 gerações novas por execução. São 30 combinações
(30.000 gerações se toda a grade for solicitada separadamente). A auditoria
aleatória usa p=0,2 (probabilidade, não número fixo de auditorias); cinco rodadas
de teste ocorrem a cada 10 rodadas. A nota é exata, a topologia é 4-regular, com
medoide, orquestração dinâmica e EMA assimétrico (`ema_slow_up`).
Os nós em conluio se comportam como honestos nas cinco primeiras rodadas.
Com `knows_schedule` e auditoria aleatória, eles não veem quais rodadas foram auditadas.

O ruído forte já implementado no simulador é `s × (|b|+1) × 10^k`, com sinal
compartilhado `s∈{-1,+1}` e `k∈[3,6]`. Ele é aplicado à geração do mesmo nó
de referência do conluio: coordena a **resposta enviada**, sem pular a geração
local dos participantes. O controle sutil usa distância relativa 0,2 em
relação à base não nula. Não confundir estes dados sintéticos de consenso
com tráfego real.

Cada nó tem próprio processo, endpoint local, geração, semente, instância Qwen
**3-0.6B NF4** e cópia residente em CUDA. Os processos compartilham só o snapshot
fixado em disco, não um serviço central de inferência. São dez instâncias por GPU
em duas GPUs, perfil de referência RTX A6000, com no mínimo 40 GiB livres em
cada dispositivo antes de iniciar. A capacidade e o tempo reais ainda precisam
ser verificados em GPU. Este perfil não é para Qwen 8B nem SmolLM3-3B.

O arquivo histórico chamado `banco_smol.json` fornece apenas perguntas, IDs e
gabaritos; respostas dele não são acessadas nesta execução. Cada cenário inicia
com diretório obrigatório novo e sem registros de inferência. O SQLite por nó
registra as novas gerações e deduplica somente a mesma solicitação em caso de
repetição técnica ou da consulta pelo nó em conluio; não há replay entre
cenários/rodadas/execuções, nem fallback para banco. A validação exige uma
geração nova única em cada nó × rodada (1.000/1.000), as 50 rodadas completas e
coerência das respostas finais com as métricas. A configuração reaproveita o
worker e o `/health` da Lara (commit `58f1f998f707ea484ab83370802c663baea18d82`),
verificando identidade, uma instância ativa e NF4 antes da coordenação.
As chamadas têm limite configurado de 600 s no protocolo; a carga de cada nó
tem limite de 5 min. O processo supervisor termina os nós ao concluir/falhar.

## Preflight e execução

Da raiz do repositório, em ambiente com as dependências node/orchestrator:

```sh
python -m infra.runpod.n20.benchmark preflight
python -m unittest infra.runpod.n20.test_benchmark sistema.tests.test_dedicated_inference sistema.tests.test_dedicated_orchestrator sistema.tests.test_runner_per_node
terraform -chdir=infra/runpod/n20 init -backend=false
terraform -chdir=infra/runpod/n20 validate
terraform -chdir=infra/runpod/n20 plan -refresh=false
```

O push nesta branch só permite teste estático/mocado sem chaves ou GPU.
A execução real do novo workflow exige disparo manual, `acknowledge_live`,
`TCC_LIVE_N20_ENABLED=true`, ambiente `tcc-live-n20` e runner Linux exclusivo
com o label `tcc-n20-cuda`. Antes de habilitar, configure revisores obrigatórios
e restrição da branch no ambiente GitHub. Instale as dependências de execução,
inclusive PyTorch CUDA e as de `sistema/node/requirements-dedicated.txt`, no runner
existente, configure `TCC_LIVE_N20_SNAPSHOT` com o snapshot local fixado pelo
perfil e deixe os endpoints 18100–18119 livres. O workflow guarda artefatos
inclusive em falhas, tenta interromper com limite interno de 110 minutos e
aplica limite do job de 120 minutos. Não usa segredos de alocação nem comandos
`terraform apply`.

GitHub só oferece `workflow_dispatch` se o arquivo do workflow existir também
na branch padrão. Até isso ser aprovado e publicado lá, o GitHub executa o
preflight no push; para execução ao vivo, após aprovação, use a CLI no próprio
runner com o mesmo perfil:

```sh
TCC_LIVE_N20_APPROVED=true python -m infra.runpod.n20.benchmark run \
  --scenario m80-alto --seed 0 --snapshot "$TCC_LIVE_N20_SNAPSHOT" \
  --output /caminho/novo/live-n20-m80-alto-s0
```

O antigo workflow RunPod não deve ser usado para iniciar este perfil. Para
provisionar N20 mais tarde, é necessário validar uma imagem compatível com
este executor e projetar seu próprio perfil de Controller/RunPod, verificação
de capacidade, custo, prazo, upload e encerramento automático. Nenhum GPU
ou serviço pago foi ativado por esta configuração.
