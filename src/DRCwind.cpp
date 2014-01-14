#include "DRCwind.h"
#include "ui_DRCwind.h"

#include "kernel/init.h"
#include "dialogs/SelectDevice.h"

DRCwind::DRCwind(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DRCwind)
{
    ui->setupUi(this);
    ui->errorList->hide();

    ui->actionOpen->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    ui->actionRunDRC->setIcon(style()->standardIcon(QStyle::SP_CommandLink));
}

DRCwind::~DRCwind()
{
    delete ui;
}

void DRCwind::on_actionOpen_triggered()
{

}

void DRCwind::on_actionRunDRC_triggered()
{
    SelectDevice d;
    if(d.exec() == QDialog::Accepted) {
        int dev = d.device();
        if(d.device() == d.deviceCPU()) {
            //kernel_main_cpu();
        } else {
            kernel_main_cuda(dev);
        }
    }
}
