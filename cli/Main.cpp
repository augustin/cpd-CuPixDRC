#include <QCoreApplication>
#include <QString>
#include <QStringList>

#include <stdio.h>

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "../kernel/init.h"
#include "../kernel/errors.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    printf("CUDA Pixel-Based Design Rule Checker\n");
    printf("    Version 0.1.1, (C) 2013-2014 Augustin Cavalier\n");
    printf("    Released under the MIT license.\n\n");

    int blocks = -1;
    int threads = -1;
    int width = -1;
    int height = -1;
    int device = -1;

    QStringList args = app.arguments();
    for(int i = 0; i < args.size(); i++) {
        QString arg = args.at(i);
        if(arg.startsWith("-b") || arg.startsWith("--blocks")) {
            blocks = args.at(i+1).toInt();
            i++;
        } else if(arg.startsWith("-t") || arg.startsWith("--threads")) {
            threads = args.at(i+1).toInt();
            i++;
        } else if(arg.startsWith("-w") || arg.startsWith("--width")) {
            width = args.at(i+1).toInt();
            i++;
        } else if(arg.startsWith("-h") || arg.startsWith("--height")) {
            height = args.at(i+1).toInt();
            i++;
        }
#ifdef CUDA
        else if(arg.startsWith("-d") || arg.startsWith("--device")) {
            device = args.at(i+1).toInt();
            i++;
        } else if(arg.startsWith("-ld") || arg.startsWith("--listdev")) {
            printf("ID\tName\t\t\tSMPs\tClock\n");
            int count = 0;
            cudaGetDeviceCount(&count);
            for(int i = 0; i < count; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                printf("%d\t", i);
                printf("%s\t", prop.name);
                printf("%s", qPrintable(QString::number(prop.multiProcessorCount)));
                printf("%s", qPrintable(QString("\t%1 GHz\n").arg(prop.clockRate/(1000.0*1000.0))));
            }

            exit(0);
        }
#endif
    }

    if(blocks == -1 || threads == -1 || width == -1 || height == -1 || device == -1) {
        printf("Arguments:\n");
        printf("    -b, --blocks\t Number of blocks to use.\n");
        printf("    -t, --threads\t Number of threads to use.\n\n");
        printf("    -w, --width\t Width of the generated testcase.\n");
        printf("    -h, --height\t Height of the generated testcase.\n\n");
#ifdef CUDA
        printf("    -d, --device\t Device to execute on.\n");
        printf("    -ld, --listdev\t List available CUDA devices.\n");
#endif
    } else {
        int* errors = 0;

        char* data;
        data = (char*)malloc(width*height);

        memset((void*)data, 'x', width*height);
        int x = 6; int y = 6;
        data[width*y+x] = ' ';
        data[width*y+x+1] = ' ';
        data[width*y+x+2] = ' ';
        data[width*y+x+3] = ' ';
        data[width*(y+1)+x] = ' ';
        data[width*(y+2)+x] = ' ';
        data[width*(y+3)+x] = ' ';

#ifdef CUDA
        errors = kernel_main_cuda(device, data, width, height, blocks, threads);
#else
        errors = kernel_main_cpu(data, width, height);
#endif
        if(errors) {
            int at = 0;
            while(at < MAX_ERRORS*3) {
                if(errors[at] == E_UNDEFINED) {
                    at += 3;
                    continue;
                }

                QStringList errorData;
                errorData.append(QString("%1, %2:").arg(errors[at+1]).arg(errors[at+2]));

                switch(errors[at]) {

                case I_THREAD_ID:
                    errorData.prepend("Info:");
                    errorData.append("x=thread ID, y=total threads");
                    break;

                case E_HOR_SPACING_TOO_SMALL:
                    errorData.prepend("Error:");
                    errorData.append("Horizontal spacing too small");
                    break;
                case E_VER_SPACING_TOO_SMALL:
                    errorData.prepend("Error:");
                    errorData.append("Vertical spacing too small");
                    break;

                default:
                    errorData.prepend("Info:");
                    errorData.append(QString("Unknown, 0x%1").arg(QString::number(errors[at], 16).toUpper()));
                    break;
                }

                printf(qPrintable(errorData.join(" ") + "\n"));
                at += 3;
            }
            free(errors);
        }
    }

    return 0;
}
