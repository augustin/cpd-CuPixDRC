#include "DRCwind.h"
#include "ui_DRCwind.h"

#include "kernel/main.h"

DRCwind::DRCwind(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::DRCwind)
{
	ui->setupUi(this);
}

DRCwind::~DRCwind()
{
	delete ui;
}

void DRCwind::on_actionOpen_triggered()
{
	kernel_main();
}
