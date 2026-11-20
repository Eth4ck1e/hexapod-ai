// Host harness for neural_core.h: reads lines "P phase cmd0..cmd8" or "S dh dw"
// from stdin and prints the 18 joint targets. Driven by tools/check_neural_core.py.
#include <stdio.h>
#include <stdlib.h>
#include "../arduino/hexapod_firmware/neural_core.h"

int main(void) {
  char kind; float v[10]; float obs[21], j[18];
  while (scanf(" %c", &kind) == 1) {
    if (kind == 'P') {
      for (int i = 0; i < 10; ++i) if (scanf("%f", &v[i]) != 1) return 1;
      nc_build_obs(v[0], v + 1, obs);
      nc_policy(obs, j);
    } else if (kind == 'S') {
      if (scanf("%f %f", &v[0], &v[1]) != 2) return 1;
      nc_stance(v[0], v[1], j);
    } else return 2;
    for (int i = 0; i < 18; ++i) printf("%.7f%c", j[i], i == 17 ? '\n' : ' ');
  }
  return 0;
}
