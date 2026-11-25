# Soup recipe catalogue: notice

`soup-recipes.json` is made from the recipe catalogue of **Soup** (<https://github.com/MakazhanAlpamys/Soup>),
licensed under the Apache License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>).

- Files read: `src/soup_cli/recipes/catalog.py` and `tests/fixtures/recipe_base_architectures.json`, at the commit
  named in the file's `source.commit`.
- What was done: the recipes (names, base models, tasks, sizes, tags, descriptions and `soup.yaml` configs) and the base
  models' architectures were read as text by `am2strata-workshop library import` and written as JSON. Nothing of Soup
  runs in Prastara; the Workshop adds each recipe's abilities and whether it trains here.
- Changes: none to the recipes' content; Soup's own marks of recipes it does not offer are kept as `unavailable`.

Copyright belongs to the Soup authors. Prastara's own recipes are in `prastara-recipes.json` (AGPL-3.0-only).
