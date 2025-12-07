"""Test authentication with wos-session cookie"""
import json
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AuthSettings

session = get_session()

# Create auth settings
auth = AuthSettings(
    source_name='fal.ai',
    cookies=json.dumps({
        'wos-session': 'Fe26.2*1*0fbdfe67be011c5fd683c4e388f1b20060ffedf074a1bdf62e48c1f22d53ca5f*OaUg4nCKbCDeXtP9jLsjxg*-fdDDay7tA3SE892oMsKDdstrIfz_sKzpnU6gN23oWAP1sq9AWiEtoVt8vqWoigolh5jZ_Y5OoeNaWbMunE85Vqf6AdEMUebq9l4OnC6evItlR1ayaaFzOWFNGFmuKL7kwtPRibg2rf66ceVWYoU-vDS1YCR-rC5_QdrG4h9bhWk8HQlUy1_JldAAv328tbQPpF-uy6LTzKsvfwHq-XJgc6TLxuR5pFoE38hvonbPkR2-LijoHbpzdG-qTrUb9EfRQi51ZBdk6wrxdgG4XIM7g7QzhBY2SfxkpBxc9uNDl04o5rC7MdZb_sMl38rcYUzavXn_BrrKQzV6IzZBIbF1SBfDZeoHVL8tdglwygDhwiHiSpgw1XCWz4vsTyX_kAEOkXvgj0GRzHAM7NDspc2h0QGwMc02369i8U5XNycuIzmoBAJgHFCDE8YLJSTa6OQzUbnSrn5nGea_tJjENxPQ0Jdr0BIcOwwveWYHvOsyLVEUT-Om1ke6ajLMoJid696l2CZ03ftTDNN5er9KwRGd21g3UVNv3-GnDOICj_QUDxpV8MDz0WTHtPAsexnZ9I9W36DzB7BwBjnUgTDxbCtTDMjwwvz7g4WPO4sVn4Uf1bvV5Ezc44ZRuQigU_56LXPnX_0mNkI8HkhMAnmDVDTF1CAEXc41avrzVHmwHfBdrDERIF1TvRpge1APVgdRaHD5UhnjEsvAVYMZfifJKdvEY8pUlR-GVzdE_zfzG1fdgkWImmfqI8r8-0Lj93RnTMxm8MEICcWDjWylafhnrgp0nozjMVrR09ASq5nto_k5syVarStx2Apszpmus_D9zRALj9asy61XXrGjtcGlZCEwXN3TD0cPKs8xveFTyxqA1gb0YEgaVCM4aNsD_JSFx2O32WhyKeVtmJ8EOyjvIgDvbg7Bqh6XDXaCfaKoaLrKzFd5a_dLS6nIbQiNlOCT0aKUwe3bcy9rheNZiyOHwpty6tI8A9rFEC552Z4Qf4LzrGEdiZWY58wwxelOOgRbH70RtDoK9-k-uRh7r5fwe7rXxKOK5KdcWQITTOSYi4HnBx4SvsTtecpzcnfIff-IRyR_hrHKPGM2IWphlv6Z38TUubE0Z2f-eedJtd0J9eMA30PXfPg3shMCF_uev_jWEZ8sU_xdBQ3SJtAKztgK8HeBWvTiyAeniR1cf5u9Z3SpDA1eGmIzlz6uRP21ZNgjt9iU28g33pxa1F8xlO9MCRs5_ei4Ui3sRpCCJfvvKDQ7YJSZWF0WalWF6kUi2NAbb4D8MkUz50D_gdhQXS0zJnsXDUnT8CxwSSROQp29B8dyeLYyXH9BuDTc2wwMPbB0hGyMcL1h82owfqzbEr5HN4-mNWydoGrIhOPvSIsfPPqf1gop0Aa8fJpogAkeM2HHed2nqXnDWIagQMx4lgHn6Uzqe4A61-kfdmhAt2f2TAI4d1wzjXqtMbpKGvRY62C8w_qobzBxeIdh8IERI69jqCLzrUAC9I4kSDVwwZMQ7J_KpWND6mgvxpVULoWG3NHnWjUpH7MTGZdeP9qsV3K09GZwQ**27a4b578d075646121e734ad05b5e7f0eec30a1ba0a34c1dfb24a5fee943bbdc*KRVOQifKQfamdVTS3H3rjSIENLEqmsh3Ols0u91HwWY~2'
    }),
    is_active=True,
    notes='Test authentication'
)

# Check if exists
existing = session.query(AuthSettings).filter_by(source_name='fal.ai').first()
if existing:
    print('Updating existing auth...')
    existing.cookies = auth.cookies
    existing.is_active = True
    existing.notes = auth.notes
else:
    print('Creating new auth...')
    session.add(auth)

session.commit()
print('✅ Auth settings saved to database!')

# Test fetching playground data with auth
print('\nTesting playground fetch with auth...')
from extractors.fal_utils.playground_fetcher import fetch_playground_data

model_id = 'fal-ai/animatediff-sparsectrl-lcm'
result = fetch_playground_data(model_id)

if result:
    print(f'✅ Got playground data:')
    print(f'   Endpoint: {result.get("endpoint")}')
    print(f'   Billing Unit: {result.get("billing_unit")}')
    print(f'   Price: ${result.get("price")}')
    print(f'   Pricing Text: {result.get("pricing_text")}')
else:
    print('❌ Failed to get playground data')

session.close()
