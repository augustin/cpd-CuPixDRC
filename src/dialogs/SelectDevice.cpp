#include "SelectDevice.h"
#include "ui_SelectDevice.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <QRadioButton>
#include <QTreeWidgetItem>

SelectDevice::SelectDevice(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::SelectDevice)
{
	ui->setupUi(this);

	/* Create Device List */
	int count = 0;
	cudaGetDeviceCount(&count);
	for(int i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		QTreeWidgetItem* itm = new QTreeWidgetItem;
		itm->setText(0, prop.name);
		itm->setText(1, QString::number(prop.multiProcessorCount));
		itm->setText(2, QString("%1 GHz").arg(prop.clockRate/(1000.0*1000.0)));
		ui->deviceList->addTopLevelItem(itm);
	}

	resizeCols();
}

SelectDevice::~SelectDevice()
{
	delete ui;
}

void SelectDevice::resizeCols()
{
	for(int i = 0; i < ui->deviceList->columnCount(); i++) {
		ui->deviceList->resizeColumnToContents(i);
	}
}

int SelectDevice::device()
{
	return ui->deviceList->indexOfTopLevelItem(ui->deviceList->currentItem());
}
