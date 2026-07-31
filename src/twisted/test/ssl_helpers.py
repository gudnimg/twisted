# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.

"""
Shared TLS and certificate fixtures for Twisted's test suite.

They are in a separate module so they will not prevent test modules importing
if pyOpenSSL is unavailable.
"""
from __future__ import annotations

import datetime
import ipaddress
import itertools
from dataclasses import dataclass
from functools import cache

from zope.interface import implementer

from OpenSSL import SSL
from OpenSSL.crypto import TYPE_RSA, X509, PKey

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    generate_private_key,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from cryptography.x509.oid import NameOID

from twisted.internet import _sslverify as sslverify, ssl
from twisted.internet.interfaces import IOpenSSLContextFactory
from twisted.python.filepath import FilePath

certPath = FilePath(__file__).sibling("server.pem").path


def _counter(counter: itertools.count[int] = itertools.count()) -> int:
    """
    Return the next integer in the natural numbers.
    """
    return next(counter)


@cache
def _keyPair(uniqueName: bytes) -> sslverify.KeyPair:
    """
    Return the cached RSA-2048 key pair identified by C{uniqueName}.
    """
    return sslverify.KeyPair.generate(kind=TYPE_RSA, size=2048)


@cache
def _certificateData(keyPair: sslverify.KeyPair, **fields: str | bytes) -> bytes:
    """
    Create and cache certificate data.
    """
    distinguishedName = sslverify.DistinguishedName(**fields)
    certificateRequest = keyPair.requestObject(distinguishedName)
    certificate = keyPair.signRequestObject(
        distinguishedName,
        certificateRequest,
        _counter(),
    )
    return certificate.dump()  # type: ignore[no-any-return]


def makeCertificate(**fields: str | bytes) -> tuple[PKey, X509]:
    """
    Return a private key and self-signed certificate for C{fields}.
    """
    distinguishedName = sslverify.DistinguishedName(**fields)
    keyPair = _keyPair(distinguishedName.commonName)
    certificate = keyPair.newCertificate(_certificateData(keyPair, **fields))
    return keyPair.original, certificate.original


def _generatePrivateKey() -> RSAPrivateKey:
    """
    Generate an RSA-4096 private key.
    """
    return generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )


_oneDay = datetime.timedelta(1)


@dataclass(eq=False, frozen=True)
class TestingAuthority:
    """
    A certificate authority used by the test suite.
    """

    name: x509.Name
    cert: x509.Certificate
    key: RSAPrivateKey

    @classmethod
    def _create(cls) -> TestingAuthority:
        """
        Create a certificate authority with fresh private key material.
        """
        aroundTimestamp = datetime.datetime.today()
        commonNameForCA = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "Testing Example CA")]
        )
        privateKeyForCA = _generatePrivateKey()
        caCertificate = (
            x509.CertificateBuilder()
            .subject_name(commonNameForCA)
            .issuer_name(commonNameForCA)
            .not_valid_before(aroundTimestamp - _oneDay)
            .not_valid_after(aroundTimestamp + _oneDay)
            .serial_number(x509.random_serial_number())
            .public_key(privateKeyForCA.public_key())
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=9),
                critical=True,
            )
            .sign(
                private_key=privateKeyForCA,
                algorithm=hashes.SHA256(),
            )
        )
        return cls(commonNameForCA, caCertificate, privateKeyForCA)

    @cache
    def _leafPrivateKey(
        self, commonName: str, *serviceIdentities: str
    ) -> RSAPrivateKey:
        """
        Return the cached private key for a leaf certificate identity.
        """
        return _generatePrivateKey()

    def issue(
        self, commonName: str, *serviceIdentities: str
    ) -> sslverify.PrivateCertificate:
        """
        Issue a certificate for one or more service identities.
        """
        privateKey = self._leafPrivateKey(commonName, *serviceIdentities)
        commonNameForCertificate = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, commonName)]
        )

        subjectAlternativeNames: list[x509.IPAddress | x509.DNSName] = []
        for serviceIdentity in serviceIdentities:
            try:
                ipAddress = ipaddress.ip_address(serviceIdentity)
            except ValueError:
                subjectAlternativeNames.append(
                    x509.DNSName(serviceIdentity.encode("idna").decode("ascii"))
                )
            else:
                subjectAlternativeNames.append(x509.IPAddress(ipAddress))

        certificateBuilder = (
            x509.CertificateBuilder()
            .subject_name(commonNameForCertificate)
            .not_valid_before(datetime.datetime.today() - _oneDay)
            .not_valid_after(datetime.datetime.today() + _oneDay)
            .serial_number(x509.random_serial_number())
            .public_key(privateKey.public_key())
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName(subjectAlternativeNames),
                critical=True,
            )
        )
        signedX509 = self._sign(certificateBuilder)
        privateCertificate: sslverify.PrivateCertificate = (
            sslverify.PrivateCertificate.loadPEM(
                b"\n".join(
                    [
                        privateKey.private_bytes(
                            Encoding.PEM,
                            PrivateFormat.TraditionalOpenSSL,
                            NoEncryption(),
                        ),
                        signedX509.public_bytes(Encoding.PEM),
                    ]
                )
            )
        )
        return privateCertificate

    def authorityCertificate(self) -> sslverify.Certificate:
        """
        Return the public certificate for this authority.
        """
        return sslverify.Certificate.loadPEM(self.cert.public_bytes(Encoding.PEM))

    def _sign(self, builder: x509.CertificateBuilder) -> x509.Certificate:
        """
        Sign a certificate using this authority's private key.
        """
        return builder.issuer_name(self.name).sign(
            private_key=self.key,
            algorithm=hashes.SHA256(),
        )


class TestingCertificateFactory:
    """
    Provide named, cached certificate authorities for tests.
    """

    @cache
    def authority(self, name: str) -> TestingAuthority:
        """
        Return the cached certificate authority identified by C{name}.
        """
        return TestingAuthority._create()

    def authorityAndServer(
        self, serviceIdentity: str = "example.com"
    ) -> tuple[sslverify.Certificate, sslverify.PrivateCertificate]:
        """
        Return the server authority certificate and a certificate issued for
        C{serviceIdentity}.
        """
        serverAuthority = self.authority("server")
        return (
            serverAuthority.authorityCertificate(),
            serverAuthority.issue("Testing Example Server", serviceIdentity),
        )


testingCertificates = TestingCertificateFactory()


class ClientTLSContext(ssl.ClientContextFactory):
    """
    SSL Context Factory for client-side connections.
    """

    isClient = 1

    def getContext(self) -> SSL.Context:
        """
        Return an L{SSL.Context} to be use for client-side connections.

        Will not return a cached context.
        This is done to improve the test coverage as most implementation
        are caching the context.
        """
        return SSL.Context(SSL.SSLv23_METHOD)


@implementer(IOpenSSLContextFactory)
class ServerTLSContext:
    """
    SSL Context Factory for server-side connections.
    """

    isClient = 0

    def __init__(
        self, filename: str | bytes = certPath, method: int | None = None
    ) -> None:
        self.filename = filename
        if method is None:
            method = SSL.SSLv23_METHOD

        self._method = method

    def getContext(self) -> SSL.Context:
        """
        Return an L{SSL.Context} to be use for server-side connections.

        Will not return a cached context.
        This is done to improve the test coverage as most implementation
        are caching the context.
        """
        ctx = SSL.Context(self._method)
        ctx.use_certificate_file(self.filename)
        ctx.use_privatekey_file(self.filename)
        return ctx
