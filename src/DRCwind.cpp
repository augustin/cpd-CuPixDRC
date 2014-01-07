#include "DRCwind.h"
#include "ui_DRCwind.h"

#include "kernel/kernel.h"

#include <QInputDialog>

DRCwind::DRCwind(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DRCwind)
{
    ui->setupUi(this);
    ui->errorList->hide();

    ui->actionOpen->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    ui->actionRunDRC->setIcon(style()->standardIcon(QStyle::SP_CommandLink));

    /* Create Device List */
    int count = 0;
    cudaGetDeviceCount(&count);
    for(int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        devices.append(prop.name);
    }
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
    int dev = devices.indexOf(QInputDialog::getItem(this, tr("Which device?"), tr("Execute DRC on:"), devices, 0, false));
    kernel_main(dev);
}
