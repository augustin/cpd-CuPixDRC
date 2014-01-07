#include "ErrorList.h"
#include "ui_ErrorList.h"

ErrorList::ErrorList(QWidget *parent) :
	QDockWidget(parent),
	ui(new Ui::ErrorList)
{
	ui->setupUi(this);

    ui->actionErrors->setIcon(style()->standardIcon(QStyle::SP_MessageBoxCritical));
    ui->actionWarnings->setIcon(style()->standardIcon(QStyle::SP_MessageBoxWarning));
    ui->actionInformation->setIcon(style()->standardIcon(QStyle::SP_MessageBoxInformation));

    resizeCols();
}

ErrorList::~ErrorList()
{
	delete ui;
}

void ErrorList::resizeCols()
{
    for(int i = 0; i < ui->list->columnCount(); i++) {
        ui->list->resizeColumnToContents(i);
    }
}

void ErrorList::addItem(QString msg)
{
    ui->list->addTopLevelItem(new QTreeWidgetItem(QStringList(msg)));
    resizeCols();
}
