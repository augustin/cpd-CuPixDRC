#include "ErrorList.h"
#include "ui_ErrorList.h"

#include "../kernel/errors.h"

ErrorList::ErrorList(QWidget *parent) :
    QDockWidget(parent),
    ui(new Ui::ErrorList)
{
    ui->setupUi(this);

    error = style()->standardIcon(QStyle::SP_MessageBoxCritical);
    warning = style()->standardIcon(QStyle::SP_MessageBoxWarning);
    information = style()->standardIcon(QStyle::SP_MessageBoxInformation);

    ui->actionErrors->setIcon(error);
    ui->actionWarnings->setIcon(warning);
    ui->actionInformation->setIcon(information);

    connect(ui->actionErrors, SIGNAL(triggered()), this, SLOT(updateFilters()));
    connect(ui->actionWarnings, SIGNAL(triggered()), this, SLOT(updateFilters()));
    connect(ui->actionInformation, SIGNAL(triggered()), this, SLOT(updateFilters()));

    resizeCols();
}

ErrorList::~ErrorList()
{
    delete ui;
}

void ErrorList::updateFilters()
{
    bool showErrors = ui->actionErrors->isChecked(),
            showWarnings = ui->actionWarnings->isChecked(),
            showInformation = ui->actionInformation->isChecked();

    for(int i = 0; i < ui->list->topLevelItemCount(); i++) {
        QTreeWidgetItem* itm = ui->list->topLevelItem(i);
        if(itm->text(0) == tr("Error")) {
            itm->setHidden(!showErrors);
        } else if(itm->text(0) == tr("Warn")) {
            itm->setHidden(!showWarnings);
        } else if(itm->text(0) == tr("Info")) {
            itm->setHidden(!showInformation);
        }
    }
}

void ErrorList::resizeCols()
{
    for(int i = 0; i < ui->list->columnCount(); i++) {
        ui->list->resizeColumnToContents(i);
    }
}

#define INFO i->setIcon(0, information); \
    i->setText(0, tr("Info")); \
    infoCnt++

#define ERR i->setIcon(0, error); \
    i->setText(0, tr("Error")); \
    errorCnt++

void ErrorList::setErrors(int* errors)
{
    ui->list->clear();

    int errorCnt = 0;
    int warnCnt = 0;
    int infoCnt = 0;

    int at = 0;
    while(at < MAX_ERRORS*3) {
        if(errors[at] == E_UNDEFINED) {
            at += 3;
            continue;
        }

        QTreeWidgetItem* i = new QTreeWidgetItem(ui->list);
        i->setText(1, QString::number(errors[at+1]));
        i->setText(2, QString::number(errors[at+2]));

        switch(errors[at]) {

        case I_THREAD_ID:
            INFO;
            i->setText(3, tr("x=thread ID, y=total threads"));
            break;

        case E_HOR_SPACING_TOO_SMALL:
            ERR;
            i->setText(3, tr("Horizontal spacing too small"));
            break;
        case E_VER_SPACING_TOO_SMALL:
            ERR;
            i->setText(3, tr("Vertical spacing too small"));
            break;

        default:
            INFO;
			i->setText(3, tr("Unknown, 0x%1").arg(QString::number(errors[at], 16).toUpper()));
            break;
        }

        at += 3;
    }

    ui->actionErrors->setText(QString("Errors (%1)").arg(errorCnt));
    ui->actionWarnings->setText(QString("Warnings (%1)").arg(warnCnt));
    ui->actionInformation->setText(QString("Info (%1)").arg(infoCnt));
    resizeCols();
}
