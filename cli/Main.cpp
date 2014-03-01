#include <QCoreApplication>
#include <QString>
#include <QStringList>
#include <QFile>

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
    qDebug("CUDA Pixel-Based Design Rule Checker");
    qDebug("    Version 0.1.2, (C) 2013-2014 Augustin Cavalier");
    qDebug("    Released under the MIT license.\n");

    int blocks = 64;
    int threads = 32;
    int width = -1;
    int height = -1;
    int device = 0;
    bool testcase = true;
    QString fileName;

    QStringList args = app.arguments();
    for(int i = 0; i < args.size(); i++) {
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
            i += 2;
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

            exit(0);
        }
#endif
    }

    if(width == -1 || height == -1) {
        printf("Commands:\n");
        printf("     testcase\t\t Generate a testcase and DRC it [default].\n");
        printf("     textchip [file]\t Load a text chip and DRC it.\n\n");
        printf("Arguments:\n");
        printf("    -w, --width\t\t Width of the chip (or testcase).\n");
        printf("    -h, --height\t Height of the chip (or testcase).\n");
#ifdef CUDA
        printf("    -b, --blocks\t Number of blocks to use [default=64].\n");
        printf("    -t, --threads\t Number of threads to use [default=32].\n");
        printf("    -d, --device\t Device to execute on [default=0].\n");
        printf("    -ld, --listdev\t List available CUDA devices.\n");
#endif
        exit(0);
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
            qDebug("The chip is empty!");
            exit(1);
        }
        qDebug("Executing DRC on device %d [%d blk, %d thrd] using chip %s...",
               device, blocks, threads, fileName.toUtf8().constData());
    }

    int* errors = 0;
#ifdef CUDA
    errors = kernel_main_cuda(device, data.constData(), width, height, blocks, threads);
#else
    errors = kernel_main_cpu(data.constData(), width, height);
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

    return 0;
}
