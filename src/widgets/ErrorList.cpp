#include "ErrorList.h"
#include "ui_ErrorList.h"

#include "../kernel/errors.h"

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

void ErrorList::setErrors(int* errors)
{
	ui->list->clear();

	int at = 0;
	while(errors[at] != E_UNDEFINED) {
		QTreeWidgetItem* i = new QTreeWidgetItem(ui->list);
		i->setText(1, QString::number(errors[at+1]));
		i->setText(2, QString::number(errors[at+2]));

		switch(errors[at]) {

		case E_HOR_SPACING_TOO_SMALL:
			i->setIcon(0, style()->standardIcon(QStyle::SP_MessageBoxCritical));
			i->setText(3, tr("Horizontal spacing too small"));
			break;
		case E_VER_SPACING_TOO_SMALL:
			i->setIcon(0, style()->standardIcon(QStyle::SP_MessageBoxCritical));
			i->setText(3, tr("Vertical spacing too small"));
			break;

		default:
			i->setIcon(0, style()->standardIcon(QStyle::SP_MessageBoxInformation));
			i->setText(3, tr("Unknown error"));
			break;
		}

		at += 3;
	}

	resizeCols();
}
