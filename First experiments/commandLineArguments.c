#include <stdio.h>
#include <stdlib.h>

#define LOG(X, Y) fprintf (fp, #X ": Time:%s, File:%s(%d) " #Y  "\n", __TIMESTAMP__, __FILE__, __LINE__)

int main(int argc, char *argv[]) {
	printf("Found %d arguments\n", argc);
  printf("The first argument is argv[0] = %s \n", argv[0]);
  printf("The second argument is argv[1] = %s \n", argv[1]);

  FILE *fp = fopen("logfile_test.txt", "w");
  if (fp == NULL) {
    printf("Error opening file!\n");
    return 1;
  }

  int i, py = 20;
  for (i = 0; i < argc; ++i){
    //LOG(WARNING, "Error encountered: check");
    fprintf(fp, "Time:%s, File:%s(%d). Integer: %d, float: %d\n", __TIMESTAMP__, __FILE__, __LINE__, i, py);
    //printf("Error encountered: check %s\n", argv[i]);
  }

  fclose(fp);
	return 0;
}
