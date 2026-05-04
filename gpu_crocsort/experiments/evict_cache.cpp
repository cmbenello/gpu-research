// Tool: fadvise DONTNEED on each file passed as arg.
// Tells the OS to release file-backed cache pages.
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        int fd = open(argv[i], O_RDONLY);
        if (fd < 0) { perror(argv[i]); continue; }
        struct stat st;
        fstat(fd, &st);
        int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        printf("%s (%.1f GB): fadvise(DONTNEED) rc=%d\n",
               argv[i], st.st_size / 1e9, rc);
        close(fd);
    }
    return 0;
}
