#define _GNU_SOURCE
// #include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <dlfcn.h>
#include <stdio.h>
#include <sys/wait.h>

typedef int (*open_t)(const char* filename, int flags, __mode_t mode);
open_t real_open;

const char* suffixes[] = {
			 ".c",
			 ".cc",
			 ".cpp",
			 ".cu",
			 ".cuh",
			 ".h",
			 ".in",
			 ".hpp",
                         ".txt",
			 NULL
};

int _open(const char* filename, int flags, __mode_t mode) {
  size_t fnlen = strlen(filename);
  if (fnlen < 4)
    return real_open(filename, flags, mode);
  int match = 0;
  for (size_t i = 0; suffixes[i] != NULL && ! match; i++) {
    size_t lensuffix = strlen(suffixes[i]);
    match = (strncmp(filename + fnlen - lensuffix, suffixes[i], lensuffix) == 0);
  }
  if (! match)
    return real_open(filename, flags, mode);

  const char* args[6];
  const char* mask = "hipifly.XXXXXX";
  char tmpfn[50];
  strcpy(tmpfn, mask);
  int res = mkstemp(tmpfn);
  if (res < 0)
    return res;
  close(res);
  args[0] = "/home/tv/pytorch/pytorch/tools/amd_build/build_amd.py";
  args[1] = "--single-file";
  args[2] = filename;
  args[3] = "--output-directory";
  args[4] = tmpfn;
  args[5] = 0;

  pid_t pid = fork();
  int status;
  if (! pid) {
    execv("python", (char* const*) args);
    _exit(1);
  } else {
    res = wait(&status);
  }
  if (WEXITSTATUS(status) == 0) {
    fprintf(stderr, "redirect open(%s, %d, %d)\n", filename, status, res);
    res = real_open(tmpfn, flags, mode);
  } else {
    res = real_open(filename, flags, mode);
  }
  unlink(tmpfn);
  return res;
}

int open (const char *filename, int flags, mode_t mode) {
  return _open(filename, flags, mode);
}

__attribute__((constructor)) static void setup(void) {
  real_open = (open_t) dlsym(RTLD_NEXT, "open"); 
  // fprintf(stderr, "called setup()\n");
}
