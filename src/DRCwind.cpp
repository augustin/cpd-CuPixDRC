#include "DRCwind.h"
#include "ui_DRCwind.h"

#include "kernel/init.h"
#include "Chip.h"
#include "ImageRequester.h"

#ifdef CUDA
#   include "dialogs/SelectDevice.h"
#endif

#include <QFileDialog>
#include <QElapsedTimer>
#include <QMessageBox>

DRCwind::DRCwind(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DRCwind)
{
    ui->setupUi(this);
    ui->errorList->hide();
    chip = 0;

#ifdef CUDA
    setWindowTitle(windowTitle() + " - CUDA");
#else
    setWindowTitle(windowTitle() + " - CPU");
#endif

    ui->actionOpen->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    ui->actionRunDRC->setIcon(style()->standardIcon(QStyle::SP_CommandLink));
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

    if(file.length()) {
        /*
        if(chip) { delete chip; }
        chip = new Chip;
        chip->load(file);
        */
        QFile f(file);
        f.open(QFile::ReadOnly);
        data = f.readAll();
        f.close();
    }
}

void DRCwind::on_actionRunDRC_triggered()
{
    int* errors = 0;
    QElapsedTimer t;

#ifdef CUDA
    SelectDevice d;
    if(d.exec() == QDialog::Accepted) {
        int dev = d.device();
        //ImageRequester* i = new ImageRequester(chip);
        t.start();
        errors = kernel_main_cuda(dev, data.constData(), 2000, 2000);
    }
#else
    //ImageRequester* i = new ImageRequester(chip);
    t.start();
    errors = kernel_main_cpu(data.constData(), 2000, 2000);
#endif

    qint64 time = t.elapsed();
    QMessageBox::information(this, tr("Total time"), tr("Took %1 ms").arg(time), QMessageBox::Ok);

    if(errors) {
        ui->errorList->setErrors(errors);
        free(errors);
        ui->errorList->show();
    }

}
