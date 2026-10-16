/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * Checks the settings page with a settings dialog made like Kdenlive's, the
 * whole way: the module loads the page (GPUFX_SETTINGS_PAGE=1 stands for
 * Kdenlive's main program), the page adds itself to the dialog, and applying
 * the dialog saves the settings and changes what the module runs on the GPU.
 *
 * Run from the bin folder of Kdenlive's Windows build, it checks the page
 * with Kdenlive's own Qt and KDE Frameworks, and Kdenlive's effect names.
 *
 * Usage: gpufx_settings_test [screenshot.png]
 */

#include "settings_page_api.h"

#include <KConfigDialog>
#include <KCoreConfigSkeleton>
#include <KPageWidgetItem>
#include <KSharedConfig>

#include <QApplication>
#include <QCheckBox>
#include <QDeadlineTimer>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QGroupBox>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QStandardPaths>
#include <QThread>

#include <framework/mlt.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char *format, ...)
{
    ++g_checks;
    if (condition) {
        return;
    }
    ++g_failures;
    va_list args;
    va_start(args, format);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

struct ModuleStatus
{
    int enabled = -1;
    QString effects;
};

ModuleStatus moduleStatus(mlt_profile profile)
{
    ModuleStatus result;
    if (mlt_filter status = mlt_factory_filter(profile, "gpufx", "noinit")) {
        result.enabled = mlt_properties_get_int(MLT_FILTER_PROPERTIES(status), "enabled");
        result.effects = QLatin1Char(',') + QString::fromUtf8(mlt_properties_get(MLT_FILTER_PROPERTIES(status), "effects")) + QLatin1Char(',');
        mlt_filter_close(status);
    }
    return result;
}

void processEvents()
{
    for (int i = 0; i < 5; ++i) {
        QCoreApplication::processEvents();
    }
}

QString settingsFileText()
{
    QFile file(QString::fromLocal8Bit(qgetenv("GPUFX_CONFIG_FILE")));
    return file.open(QIODevice::ReadOnly) ? QString::fromUtf8(file.readAll()) : QString();
}

/** Registers the module from the build folder, unless MLT loaded it */
bool registerModule(mlt_repository repository)
{
    if (mlt_properties_get_data(mlt_repository_filters(repository), "gpufx", nullptr)) {
        return true;
    }
#if defined(_WIN32)
    void *module = reinterpret_cast<void *>(LoadLibraryA(GPUFX_TEST_MODULE));
    auto symbol = module ? reinterpret_cast<void (*)()>(GetProcAddress(reinterpret_cast<HMODULE>(module), "mlt_register")) : nullptr;
#else
    void *module = dlopen(GPUFX_TEST_MODULE, RTLD_NOW | RTLD_GLOBAL);
    auto symbol = module ? reinterpret_cast<void (*)()>(dlsym(module, "mlt_register")) : nullptr;
#endif
    if (!symbol) {
        return false;
    }
    reinterpret_cast<void (*)(mlt_repository)>(symbol)(repository);
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    // Qt's test folders instead of the user's settings
    QStandardPaths::setTestModeEnabled(true);
    qputenv("GPUFX_SETTINGS_PAGE", "1");
    qunsetenv("GPUFX_CONFIG_FILE");
    // As Kdenlive does: its Windows build has MLT's Qt 5 modules too
    qputenv("MLT_REPOSITORY_DENY", "libmltqt:libmltglaxnimate");
    if (qEnvironmentVariableIsEmpty("GPUFX_CUDA_LIBRARY")) {
        qputenv("GPUFX_CUDA_LIBRARY", GPUFX_TEST_CUDA_LIBRARY);
    }
    QApplication app(argc, argv);
    // Kdenlive's name: its settings file, and its effects when they are there
    QCoreApplication::setApplicationName(QStringLiteral("kdenlive"));
    const QString settingsFile = QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation) + QStringLiteral("/kdenliverc");
    QFile::remove(settingsFile);

    mlt_repository repository = mlt_factory_init(qgetenv("MLT_REPOSITORY").isEmpty() ? nullptr : qgetenv("MLT_REPOSITORY").constData());
    if (!repository || !registerModule(repository)) {
        fprintf(stderr, "cannot load MLT or %s\n", GPUFX_TEST_MODULE);
        return 2;
    }
    mlt_profile profile = mlt_profile_init(nullptr);

    // The page tells the render processes which settings file to read
    const QString exported = QDir::fromNativeSeparators(QString::fromLocal8Bit(qgetenv("GPUFX_CONFIG_FILE")));
    printf("settings file: %s\n", qPrintable(exported));
    check(exported == settingsFile, "GPUFX_CONFIG_FILE is %s instead of %s: the page did not start", qPrintable(exported), qPrintable(settingsFile));

    // Another KConfigDialog does not get the page
    KCoreConfigSkeleton otherSettings(KSharedConfig::openConfig(QStringLiteral("otherrc")));
    auto *other = new KConfigDialog(nullptr, QStringLiteral("other"), &otherSettings);
    other->addPage(new QWidget, QStringLiteral("Other"));
    other->show();
    processEvents();
    check(!other->findChild<QWidget *>(QStringLiteral("gpufx_page")), "the page was added to a dialog that is not Kdenlive's settings");
    delete other;

    // A dialog like Kdenlive's settings
    KCoreConfigSkeleton appSettings(KSharedConfig::openConfig());
    bool appValue = false;
    appSettings.addItemBool(QStringLiteral("app_value"), appValue, false);
    auto *dialog = new KConfigDialog(nullptr, QStringLiteral("settings"), &appSettings);
    auto *misc = new QWidget;
    auto *appBox = new QCheckBox(QStringLiteral("Application setting"), misc);
    appBox->setObjectName(QStringLiteral("kcfg_app_value"));
    dialog->addPage(misc, QStringLiteral("Misc"), QStringLiteral("configure"));
    dialog->show();
    processEvents();

    auto *item = qobject_cast<KPageWidgetItem *>(dialog->property("gpufx_page").value<QObject *>());
    check(item != nullptr, "the page was not added to the settings dialog");
    auto *enabled = dialog->findChild<QCheckBox *>(QStringLiteral("kcfg_enabled"));
    auto *dropShadow = dialog->findChild<QCheckBox *>(QStringLiteral("kcfg_effect:dropshadow"));
    auto *compositing = dialog->findChild<QCheckBox *>(QStringLiteral("kcfg_transition:qtblend"));
    auto *status = dialog->findChild<QLabel *>(QStringLiteral("gpufx_status"));
    auto *buttons = dialog->findChild<QDialogButtonBox *>();
    QPushButton *apply = buttons ? buttons->button(QDialogButtonBox::Apply) : nullptr;
    QPushButton *defaults = buttons ? buttons->button(QDialogButtonBox::RestoreDefaults) : nullptr;
    if (!item || !enabled || !dropShadow || !compositing || !status || !apply || !defaults) {
        check(false, "widgets of the page or the dialog are missing");
        printf("%d checks, %d failures\n", g_checks, g_failures);
        return 1;
    }
    printf("page: %s, icon %s\n", qPrintable(item->name()), qPrintable(item->icon().name()));
    for (QCheckBox *box : dialog->findChildren<QCheckBox *>()) {
        if (box->objectName().startsWith(QLatin1String("kcfg_")) && box != appBox) {
            printf("  %s: %s%s\n", qPrintable(box->objectName().mid(5)), qPrintable(box->text()), box->isChecked() ? "" : " (off)");
            check(box->isChecked(), "%s is off by default", qPrintable(box->objectName()));
        }
    }
    if (!QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("effects/dropshadow.xml")).isEmpty()) {
        // Kdenlive's names
        check(dropShadow->text() == QLatin1String("Drop Shadow"), "dropshadow is called %s", qPrintable(dropShadow->text()));
        auto *brightness = dialog->findChild<QCheckBox *>(QStringLiteral("kcfg_effect:brightness"));
        check(brightness && brightness->text().contains(QLatin1String("Fade in")), "brightness is not called Fade in");
    }
    check(compositing->text().startsWith(QLatin1String("Track compositing")), "the qtblend transition is called %s", qPrintable(compositing->text()));
    check(!apply->isEnabled(), "Apply is enabled before any change");

    // The description of the GPU comes from a thread
    QDeadlineTimer deadline(30000);
    while (status->text().startsWith(QLatin1String("Checking")) && !deadline.hasExpired()) {
        processEvents();
        QThread::msleep(20);
    }
    printf("status: %s\n", qPrintable(status->text()));
    check(!status->text().startsWith(QLatin1String("Checking")), "the GPU status did not arrive");
    if (QString::fromLocal8Bit(qgetenv("GPUFX_CUDA_LIBRARY")).contains(QLatin1String("mockcuda"))) {
        check(status->text().contains(QLatin1String("gpufx emulated GPU")), "the status does not show the emulated GPU");
    }

    // A screenshot of the page
    if (argc > 1) {
        dialog->setCurrentPage(item);
        dialog->resize(900, 680);
        processEvents();
        check(dialog->grab().save(QString::fromLocal8Bit(argv[1])), "cannot save the screenshot");
    }

    // One effect kept on the CPU
    ModuleStatus module = moduleStatus(profile);
    check(module.enabled == 1 && module.effects.contains(QLatin1String(",dropshadow,")), "the module does not start with every effect on");
    dropShadow->setChecked(false);
    processEvents();
    check(apply->isEnabled(), "Apply is not enabled after a change");
    apply->click();
    processEvents();
    QString text = settingsFileText();
    check(text.contains(QLatin1String("[gpufx]")) && text.contains(QLatin1String("effect:dropshadow=false")), "the settings file does not keep dropshadow off:\n%s",
          qPrintable(text));
    module = moduleStatus(profile);
    check(module.enabled == 1 && !module.effects.contains(QLatin1String(",dropshadow,")) && module.effects.contains(QLatin1String(",qtblend,")),
          "the module did not take dropshadow off the GPU: %s", qPrintable(module.effects));

    // GPU processing off
    enabled->setChecked(false);
    processEvents();
    check(!dropShadow->isEnabled(), "the effect switches stay enabled with GPU processing off");
    apply->click();
    processEvents();
    check(settingsFileText().contains(QLatin1String("enabled=false")), "the settings file does not have enabled=false");
    check(moduleStatus(profile).enabled == 0, "the module did not turn GPU processing off");

    // Defaults: everything on, nothing left in the file
    defaults->click();
    processEvents();
    check(enabled->isChecked() && dropShadow->isChecked() && dropShadow->isEnabled(), "Defaults did not turn everything back on");
    apply->click();
    processEvents();
    text = settingsFileText();
    check(!text.contains(QLatin1String("enabled=false")) && !text.contains(QLatin1String("dropshadow")), "settings left after Defaults:\n%s", qPrintable(text));
    module = moduleStatus(profile);
    check(module.enabled == 1 && module.effects.contains(QLatin1String(",dropshadow,")), "the module did not go back to the defaults");

    // Opened again: the same page, once
    dialog->hide();
    processEvents();
    dialog->show();
    processEvents();
    check(dialog->findChildren<QWidget *>(QStringLiteral("gpufx_page")).size() == 1, "the page was added twice");
    delete dialog;

    printf("%d checks, %d failures\n", g_checks, g_failures);
    mlt_profile_close(profile);
    QFile::remove(settingsFile);
    return g_failures ? 1 : 0;
}
