#include "DRCwind.h"
#include "ui_DRCwind.h"

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
