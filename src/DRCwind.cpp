#include "DRCwind.h"
#include "ui_DRCwind.h"

#include "kernel/init.h"
#include "Chip.h"
#include "ImageRequester.h"

#include "dialogs/SelectDevice.h"

#include <QFileDialog>
#include <QElapsedTimer>

DRCwind::DRCwind(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DRCwind)
{
    ui->setupUi(this);
    ui->errorList->hide();
    chip = 0;

    ui->actionOpen->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    ui->actionRunDRC->setIcon(style()->standardIcon(QStyle::SP_CommandLink));
}

DRCwind::~DRCwind()
{
    delete ui;
}

void DRCwind::on_actionOpen_triggered()
{
    QString file =
            QFileDialog::getOpenFileName(this,
                                         tr("Select Chip File"), "",
                                         tr("All supported files (%1)").arg(Chip::supportedFormats().join(";;")),
                                         0);
    if(file.length()) {
        if(chip) { delete chip; }
        chip = new Chip;
        chip->load(file);
    }
}

void DRCwind::on_actionRunDRC_triggered()
{
    SelectDevice d;
    if(d.exec() == QDialog::Accepted) {
        int dev = d.device();
        ImageRequester* i = new ImageRequester(chip);

        if(d.device() == d.deviceCPU()) {
            //kernel_main_cpu();
        } else {
            kernel_main_cuda(dev);
        }
    }
}
