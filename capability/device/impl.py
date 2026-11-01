"""Runtime facade for logical hardware roles."""

from device_management import HardwareProfileManager


def get_profile(args=None):
    return HardwareProfileManager().profile()


def inventory(args=None):
    devices = HardwareProfileManager().inventory()
    return {"devices": devices, "count": len(devices)}


def status(args=None):
    return HardwareProfileManager().resolve_all()


def resolve(args):
    return HardwareProfileManager().resolve(args["role"])


ACTIONS = {
    "device.profile": get_profile,
    "device.inventory": inventory,
    "device.status": status,
    "device.resolve": resolve,
}
