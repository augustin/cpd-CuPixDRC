#include "DRCwind.h"
#include "ui_DRCwind.h"

#include "../kernel/init.h"
#include "Chip.h"

#ifdef CUDA
#   include "dialogs/SelectDevice.h"
#endif

#include <QFileDialog>
#include <QInputDialog>

DRCwind::DRCwind(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DRCwind)
{
    ui->setupUi(this);
    ui->errorList->hide();
    chip = 0;

#ifdef CUDA
    setWindowTitle(windowTitle() + " - CUDA");

    SelectDevice d;
    if(d.exec() == QDialog::Accepted) {
        cudaDevice = d.device();
    } else {
        QApplication::exit();
    }
#else
    setWindowTitle(windowTitle() + " - CPU");
#endif

    ui->actionOpen->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    ui->actionRunDRC->setIcon(style()->standardIcon(QStyle::SP_CommandLink));
    ui->actionTestcase->setIcon(style()->standardIcon(QStyle::SP_TitleBarShadeButton));
}

DRCwind::~DRCwind()
{
    delete ui;
}

void DRCwind::on_actionOpen_triggered()
{
    QStringList supportedFormats = Chip::supportedFormats();
    supportedFormats << "*.txt";

    QString file =
            QFileDialog::getOpenFileName(this,
                            tr("Select Chip File"), "",
                            tr("All supported files (%1)")
                            .arg(supportedFormats.join(";;")), 0);

    if(file.length() && file.endsWith(".txt", Qt::CaseInsensitive)) {
        QFile f(file);
        f.open(QFile::ReadOnly);
        data = f.readAll();
        f.close();

        /* CONSTANTS */
        imgW = 287;
        imgH = 697;
        //imgW = 10;
        //imgH = 10;
    } else {
        data = QByteArray();
        if(chip) { delete chip; }
        chip = new Chip;
        chip->load(file);
    }
}

void DRCwind::on_actionTestcase_triggered()
{
    imgW = QInputDialog::getInt(this, tr("Width of testcase:"), tr("Width:"), 10, 10, 32000);
    imgH = QInputDialog::getInt(this, tr("Height of testcase:"), tr("Height:"), 10, 10, 32000);

    data = QByteArray(imgW*imgH, 'x');
    int x = 6; int y = 6;
    data[imgW*y+x] = ' ';
    data[imgW*y+x+1] = ' ';
    data[imgW*y+x+2] = ' ';
    data[imgW*y+x+3] = ' ';
    data[imgW*(y+1)+x] = ' ';
    data[imgW*(y+2)+x] = ' ';
    data[imgW*(y+3)+x] = ' ';

    on_actionRunDRC_triggered();
}

void DRCwind::on_actionRunDRC_triggered()
{
    int* errors = 0;
    //QElapsedTimer t;
    //t.start();
    //qint64 time = t.elapsed();
    //QMessageBox::information(this, tr("Total time"), tr("Took %1 ms").arg(time), QMessageBox::Ok);

    if(data.size()) {
#ifdef CUDA
        errors = kernel_main_cuda(cudaDevice, data.constData(), imgW, imgH, 64, 32);
#else
        errors = kernel_main_cpu(data.constData(), imgW, imgH);
#endif
    } else {
        // TODO
    }

    if(errors) {
        ui->errorList->setErrors(errors);
        free(errors);
        ui->errorList->show();
    }
}
