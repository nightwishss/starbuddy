import unittest

from gateway.phone.onboarding import PAGE


class PhoneOnboardingTests(unittest.TestCase):
    def test_page_contains_certificate_and_https_console_links(self):
        page = PAGE.replace("{console_url}", "https://192.168.1.74:8443")
        self.assertIn("/VENTUNO-Q-Local-CA.crt", page)
        self.assertIn("https://192.168.1.74:8443", page)

    def test_page_does_not_request_secrets(self):
        self.assertNotIn("password", PAGE.lower())
        self.assertNotIn("api/pair", PAGE)


if __name__ == "__main__":
    unittest.main()
