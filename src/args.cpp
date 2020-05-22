#include "includes.h"

/* delete the ith argument */
void del_arg(int argc, char **argv, int index)
{
    int i;
    for (i = index; i < argc-1; ++i)
    {
        argv[i] = argv[i+1];
    }
    argv[i] = 0;
}

/* check if an argument exists, 1: yes, 0: no */
int find_arg(int argc, char *argv[], const char *arg)
{
    for (int i = 0; i < argc; ++i)
    {
        if (!argv[i])
        {
            continue;
        }
        if (0 == strcmp(argv[i], arg))
        {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}
