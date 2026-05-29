# Token Efficiency by Encoding Format

Average `cl100k_base` token counts per encoding format, measured on the
example graphs in `examples/` (see `examples/manifest.json`). The `visual`
format is excluded (it renders an image; its token_count is 0 by definition).

| Format | small (avg tokens) | medium (avg tokens) | large (avg tokens) |
| --- | --- | --- | --- |
| adjacency_list | 486.5 | 1062.0 | 11541.0 |
| edge_list | 546.5 | 1542.0 | 17132.0 |
| mermaid | 743.2 | 1583.0 | 17165.0 |
| dot | 857.8 | 1995.0 | 21902.0 |
| natural_language | 771.5 | 1983.0 | 21632.0 |
| matrix | 1197.8 | 1101.0 | 47788.0 |

## Graph sizes per tier (examples used)

| Tier | Graphs | avg nodes | avg edges |
| --- | --- | --- | --- |
| small | 4 | 18.5 | 45.5 |
| medium | 1 | 20.0 | 90.0 |
| large | 1 | 150.0 | 1007.0 |

## Average tokens per edge

| Format | small (tokens/edge) | medium (tokens/edge) | large (tokens/edge) |
| --- | --- | --- | --- |
| adjacency_list | 10.69 | 11.8 | 11.46 |
| edge_list | 12.01 | 17.13 | 17.01 |
| mermaid | 16.33 | 17.59 | 17.05 |
| dot | 18.85 | 22.17 | 21.75 |
| natural_language | 16.96 | 22.03 | 21.48 |
| matrix | 26.33 | 12.23 | 47.46 |

_Generated from example graphs via grb.encoder.encode_graph._
