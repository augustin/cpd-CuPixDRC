#include <QCoreApplication>
#include <QString>
#include <QStringList>
#include <QFile>
#include <QElapsedTimer>

#include <stdio.h>

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "../kernel/init.h"
#include "../kernel/errors.h"

#ifndef NO_INFO
    #define msgErr qDebug
    #define msgOut printf
#else
    #define msgErr
    #define msgOut
#endif

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

#ifdef CUDA
    msgErr("[CUDA] Pixel-Based Design Rule Checker");
#else
    msgErr("[CPU] Pixel-Based Design Rule Checker");
#endif
    msgErr("    Version %s, (C) 2013-2014 Augustin Cavalier", KERNEL_VERSION);
    msgErr("    Released under the MIT license.\n");

    int blocks = 64;
    int threads = 32;
    int width = -1;
    int height = -1;
#ifdef CUDA
    int device = 0;
#endif
    bool testcase = true;
    QString fileName;

    QStringList args = app.arguments();
    int i = 1;
    while(i < args.size()) {
        QString arg = args.at(i);
        if(arg == "textchip") {
            testcase = false;
            fileName = args.at(i+1);
            i++;
        } else if(arg == "-b" || arg == "--blocks") {
            blocks = args.at(i+1).toInt();
            i++;
        } else if(arg == "-t" || arg == "--threads") {
            threads = args.at(i+1).toInt();
            i++;
        } else if(arg == "-w" || arg == "--width") {
            width = args.at(i+1).toInt();
            i++;
        } else if(arg == "-h" || arg == "--height") {
            height = args.at(i+1).toInt();
            i++;
        }
#ifdef CUDA
        else if(arg == "-d" || arg == "--device") {
            device = args.at(i+1).toInt();
            i++;
        } else if(arg == "-ld" || arg == "--listdev") {
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

            exit(1);
        }
#endif
        i++;
    }

    if(width == -1 || height == -1) {
        printf("Commands:\n");
        printf("     testcase\t\t Generate a testcase and DRC it [default].\n");
        printf("     textchip [file]\t Load a text chip and DRC it.\n\n");
        printf("Arguments:\n");
        printf("    -w, --width\t\t Width of the chip (or testcase).\n");
        printf("    -h, --height\t Height of the chip (or testcase).\n");
        printf("    --batch\t Run in batch-test mode with the specified number of runs.\n");
#ifdef CUDA
        printf("    -b, --blocks\t Number of blocks to use [default=64].\n");
        printf("    -t, --threads\t Number of threads to use [default=32].\n");
        printf("    -d, --device\t Device to execute on [default=0].\n");
        printf("    -ld, --listdev\t List available CUDA devices.\n");
#endif
        exit(1);
    }

    QByteArray data;
    if(testcase) {
        data = QByteArray('x', width*height);
        int x = 6; int y = 6;
        data[width*y+x] = ' ';
        data[width*y+x+1] = ' ';
        data[width*y+x+2] = ' ';
        data[width*y+x+3] = ' ';
        data[width*(y+1)+x] = ' ';
        data[width*(y+2)+x] = ' ';
        data[width*(y+3)+x] = ' ';
    } else {
        QFile f(fileName);
        f.open(QFile::ReadOnly);
        data = f.readAll();
        f.close();
        if(!data.size()) {
            msgErr("The chip is empty!");
            exit(1);
        }
        msgErr("Using chipfile %s.", fileName.toUtf8().constData());
    }


#ifdef CUDA
    msgErr("Executing DRC on device %d [%d blk, %d thrd]...",
           device, blocks, threads);
#else
    msgErr("Executing DRC on the CPU...");
#endif

    int* errors = 0;
    msgOut("cudaMalloc/cudaMemcpy\texecute\tcudaMemcpy/cudaFree\n");
    //printf("%d", threads);
#ifdef CUDA
    errors = kernel_main_cuda(device, data.constData(), width, height, blocks, threads);
#else
    errors = kernel_main_cpu(data.constData(), width, height);
#endif
    printf("\n");

#ifndef NO_INFO
    if(errors) {
#define ERR errorData.prepend("Error:"); \
        errCnt++
#define INFO errorData.prepend("Info:"); \
        infoCnt++
#define WARN errorData.prepend("Warning:"); \
        warnCnt++

        int errCnt = 0;
        int infoCnt = 0;
        int warnCnt = 0;
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
                INFO;
                errorData.append("x=thread ID, y=total threads");
                break;

            case E_HOR_SPACING_TOO_SMALL:
                ERR;
                errorData.append("Horizontal spacing too small");
                break;
            case E_VER_SPACING_TOO_SMALL:
                ERR;
                errorData.append("Vertical spacing too small");
                break;

            default:
                INFO;
                errorData.append(QString("Unknown, 0x%1").arg(QString::number(errors[at], 16).toUpper()));
                break;
            }

            msgOut("%s", qPrintable(errorData.join(" ") + "\n"));
            at += 3;
        }
        msgErr("TOTALS: %d errors, %d warnings, %d infos.", errCnt, warnCnt, infoCnt);
    }
#endif

    free(errors);
    return 0;
}
